use std::sync::Arc;

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::{
    CudaContext, CudaEvent, CudaStream, DeviceRepr, PinnedHostSlice, ValidAsZeroBits,
    sys::CUevent_flags,
};
use half::bf16;

use super::{blaslt::BlasLt, kernels::Kernels};

use crate::tensor::{BufferPool, BufferPoolStats, Shape, Tensor};

const BF16_POOL_MAX_AVAILABLE_ELEMENTS: usize = 64 * 1024 * 1024;
const FP8_POOL_MAX_AVAILABLE_ELEMENTS: usize = 32 * 1024 * 1024;

fn cuda_graphs_enabled_from_env() -> bool {
    std::env::var("LFM25_CUDA_GRAPHS")
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(false)
}

pub struct CudaRuntime {
    _context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernels: Kernels,
    blaslt: BlasLt,
    bf16_pool: Arc<BufferPool<bf16>>,
    fp8_pool: Arc<BufferPool<u8>>,
    u32_pool: Arc<BufferPool<u32>>,
    graph_capture_compatible: bool,
}

pub(crate) struct TimingEvent {
    event: CudaEvent,
}

impl CudaRuntime {
    pub fn new(device: usize) -> Result<Self> {
        let context = CudaContext::new(device)
            .with_context(|| format!("failed to create CUDA context on device {device}"))?;
        let graph_capture_compatible = cuda_graphs_enabled_from_env();
        if graph_capture_compatible {
            // This runtime owns exactly one compute stream. cudarc's automatic
            // cross-stream event tracking creates dependencies on pre-capture
            // events, which CUDA Graph stream capture rejects. Disable it before
            // any device allocation is created when graph mode is explicitly
            // requested. Single-stream ordering remains the synchronization
            // contract for the graph-enabled runtime.
            unsafe {
                context.disable_event_tracking();
            }
        }

        let stream = context
            .new_stream()
            .context("failed to create CUDA compute stream")?;

        let kernels = Kernels::load(&context).context("faled to load CUDA kernels")?;

        let blaslt = BlasLt::new(stream.clone())?;

        Ok(Self {
            _context: context,
            stream,
            kernels,
            blaslt,
            bf16_pool: Arc::new(BufferPool::new(BF16_POOL_MAX_AVAILABLE_ELEMENTS)),
            fp8_pool: Arc::new(BufferPool::new(FP8_POOL_MAX_AVAILABLE_ELEMENTS)),
            u32_pool: Arc::new(BufferPool::new(1024 * 1024)),
            graph_capture_compatible,
        })
    }

    #[cfg(test)]
    pub(crate) fn context(&self) -> &Arc<CudaContext> {
        &self._context
    }

    pub(crate) fn graph_capture_compatible(&self) -> bool {
        self.graph_capture_compatible
    }

    pub(crate) fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub(crate) fn device_name(&self) -> Result<String> {
        self._context
            .name()
            .context("failed to query CUDA device name")
    }

    pub(crate) fn memory_info(&self) -> Result<(usize, usize)> {
        self._context
            .mem_get_info()
            .context("failed to query CUDA memory")
    }

    pub(crate) fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub(crate) fn blaslt(&self) -> &BlasLt {
        &self.blaslt
    }

    pub fn upload<T>(&self, src: &[T], shape: Shape) -> Result<Tensor<T>>
    where
        T: DeviceRepr,
    {
        ensure!(
            src.len() == shape.numel(),
            "upload shape mismatch: shape requires {} elements, source contains {}",
            shape.numel(),
            src.len(),
        );

        let storage = self
            .stream
            .clone_htod(src)
            .context("failed to upload tensor to GPU")?;

        Tensor::new(storage, shape)
    }

    pub(crate) fn upload_prefix<T>(&self, src: &[T], destination: &mut Tensor<T>) -> Result<()>
    where
        T: DeviceRepr,
    {
        ensure!(
            src.len() <= destination.storage_capacity(),
            "upload prefix has {} elements but destination capacity is {}",
            src.len(),
            destination.storage_capacity()
        );
        let mut view = destination
            .storage_mut()
            .try_slice_mut(0..src.len())
            .context("invalid destination prefix")?;
        self.stream
            .memcpy_htod(src, &mut view)
            .context("failed to update persistent GPU metadata")?;
        Ok(())
    }

    pub(crate) fn upload_range<T>(
        &self,
        src: &[T],
        destination: &mut Tensor<T>,
        start: usize,
    ) -> Result<()>
    where
        T: DeviceRepr,
    {
        let end = start
            .checked_add(src.len())
            .context("GPU metadata range overflow")?;
        ensure!(
            end <= destination.storage_capacity(),
            "GPU metadata range exceeds capacity"
        );
        let mut view = destination
            .storage_mut()
            .try_slice_mut(start..end)
            .context("invalid destination metadata range")?;
        self.stream
            .memcpy_htod(src, &mut view)
            .context("failed to update persistent GPU metadata range")?;
        Ok(())
    }

    pub fn zeros<T>(&self, shape: Shape) -> Result<Tensor<T>>
    where
        T: DeviceRepr + ValidAsZeroBits,
    {
        let storage = self
            .stream
            .alloc_zeros::<T>(shape.numel())
            .context("failed to allocate zeroed GPU tensor")?;

        Tensor::new(storage, shape)
    }

    /// Allocates device storage without initializing it.
    ///
    /// Callers must guarantee that the next operation writes every element
    /// before any read. Keeping this boundary in the runtime makes accidental
    /// use of uninitialized device memory visible at review time while avoiding
    /// a redundant memset for outputs that kernels overwrite completely.
    pub(crate) fn alloc_uninit<T>(&self, shape: Shape) -> Result<Tensor<T>>
    where
        T: DeviceRepr,
    {
        let storage = unsafe { self.stream.alloc::<T>(shape.numel()) }
            .context("failed to allocate uninitialized GPU tensor")?;

        Tensor::new(storage, shape)
    }

    pub(crate) fn alloc_bf16(&self, shape: Shape) -> Result<Tensor<bf16>> {
        let elements = shape.numel();
        let allocation_elements = BufferPool::<bf16>::allocation_elements(elements)
            .context("BF16 pool size class overflow")?;
        let storage = match self.bf16_pool.take(elements) {
            Some(storage) => storage,
            None => unsafe { self.stream.alloc::<bf16>(allocation_elements) }
                .context("failed to allocate pooled BF16 GPU tensor")?,
        };
        Tensor::new_pooled(storage, shape, Arc::clone(&self.bf16_pool))
    }

    pub(crate) fn alloc_fp8(&self, shape: Shape) -> Result<Tensor<u8>> {
        let elements = shape.numel();
        let allocation_elements = BufferPool::<u8>::allocation_elements(elements)
            .context("FP8 pool size class overflow")?;
        let storage = match self.fp8_pool.take(elements) {
            Some(storage) => storage,
            None => unsafe { self.stream.alloc::<u8>(allocation_elements) }
                .context("failed to allocate pooled FP8 GPU tensor")?,
        };
        Tensor::new_pooled(storage, shape, Arc::clone(&self.fp8_pool))
    }

    pub(crate) fn alloc_u32(&self, shape: Shape) -> Result<Tensor<u32>> {
        let elements = shape.numel();
        let allocation_elements = BufferPool::<u32>::allocation_elements(elements)
            .context("u32 pool size class overflow")?;
        let storage = match self.u32_pool.take(elements) {
            Some(storage) => storage,
            None => unsafe { self.stream.alloc::<u32>(allocation_elements) }
                .context("failed to allocate pooled u32 GPU tensor")?,
        };
        Tensor::new_pooled(storage, shape, Arc::clone(&self.u32_pool))
    }

    pub(crate) fn zero_bf16_range(
        &self,
        destination: &mut Tensor<bf16>,
        start: usize,
        elements: usize,
    ) -> Result<()> {
        let end = start.checked_add(elements).context("zero range overflow")?;
        ensure!(
            end <= destination.storage_capacity(),
            "zero range exceeds tensor storage"
        );
        let mut view = destination
            .storage_mut()
            .try_slice_mut(start..end)
            .context("invalid zero range")?;
        self.stream
            .memset_zeros(&mut view)
            .context("failed to zero GPU state range")
    }

    pub(crate) fn copy_bf16_range(
        &self,
        source: &Tensor<bf16>,
        source_start: usize,
        destination: &mut Tensor<bf16>,
        destination_start: usize,
        elements: usize,
    ) -> Result<()> {
        let source_end = source_start
            .checked_add(elements)
            .context("source copy range overflow")?;
        let destination_end = destination_start
            .checked_add(elements)
            .context("destination copy range overflow")?;
        ensure!(
            source_end <= source.storage_capacity(),
            "source copy range exceeds tensor storage"
        );
        ensure!(
            destination_end <= destination.storage_capacity(),
            "destination copy range exceeds tensor storage"
        );
        let source = source
            .storage()
            .try_slice(source_start..source_end)
            .context("invalid source copy range")?;
        let mut destination = destination
            .storage_mut()
            .try_slice_mut(destination_start..destination_end)
            .context("invalid destination copy range")?;
        self.stream
            .memcpy_dtod(&source, &mut destination)
            .context("failed to copy recurrent state on GPU")
    }

    pub(crate) fn pack_rows_bf16(
        &self,
        first: &Tensor<bf16>,
        second: &Tensor<bf16>,
    ) -> Result<Tensor<bf16>> {
        ensure!(
            first.rank() == 2 && second.rank() == 2,
            "row packing requires rank-2 tensors"
        );
        ensure!(
            first.dims()[1] == second.dims()[1],
            "row packing column mismatch: first={:?}, second={:?}",
            first.dims(),
            second.dims()
        );
        let rows = first.dims()[0]
            .checked_add(second.dims()[0])
            .context("packed row count overflow")?;
        let elements = first
            .numel()
            .checked_add(second.numel())
            .context("packed weight size overflow")?;
        let shape = Shape::new([rows, first.dims()[1]]);
        ensure!(shape.numel() == elements, "packed weight shape overflow");
        let mut output = self.alloc_uninit::<bf16>(shape)?;
        let split = first.numel();
        {
            let mut destination = output
                .storage_mut()
                .try_slice_mut(0..split)
                .context("invalid first packed weight range")?;
            self.stream
                .memcpy_dtod(first.storage(), &mut destination)
                .context("failed to pack first weight")?;
        }
        {
            let mut destination = output
                .storage_mut()
                .try_slice_mut(split..elements)
                .context("invalid second packed weight range")?;
            self.stream
                .memcpy_dtod(second.storage(), &mut destination)
                .context("failed to pack second weight")?;
        }
        Ok(output)
    }

    pub(crate) fn bf16_pool_stats(&self) -> BufferPoolStats {
        self.bf16_pool.stats()
    }

    pub(crate) fn fp8_pool_stats(&self) -> BufferPoolStats {
        self.fp8_pool.stats()
    }

    pub(crate) fn record_timing_event(&self) -> Result<TimingEvent> {
        let event = self
            ._context
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .context("failed to create CUDA timing event")?;
        event
            .record(&self.stream)
            .context("failed to record CUDA timing event")?;
        Ok(TimingEvent { event })
    }

    pub(crate) fn elapsed_ms(&self, start: &TimingEvent, end: &TimingEvent) -> Result<f64> {
        end.event
            .synchronize()
            .context("failed to synchronize CUDA timing event")?;
        Ok(f64::from(
            start
                .event
                .elapsed_ms(&end.event)
                .context("failed to measure CUDA event interval")?,
        ))
    }

    pub fn download<T>(&self, tensor: &Tensor<T>) -> Result<Vec<T>>
    where
        T: DeviceRepr,
    {
        let logical = tensor
            .storage()
            .try_slice(0..tensor.numel())
            .context("invalid logical tensor download range")?;
        self.stream
            .clone_dtoh(&logical)
            .context("failed to download tensor from GPU")
    }

    pub(crate) fn pinned_u32(&self, elements: usize) -> Result<PinnedHostSlice<u32>> {
        ensure!(elements > 0, "pinned output ring cannot be empty");
        // SAFETY: callers copy every logical output element from device memory
        // before reading it. Unused capacity is never observed.
        unsafe { self._context.alloc_pinned::<u32>(elements) }
            .context("failed to allocate pinned host output ring")
    }

    pub(crate) fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .context("failed to synchronize CUDA stream")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a CUDA-capable GPU"]
    fn upload_tensor() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;

        let tensor = runtime.upload(&[1.0f32, 2.0, 3.0], Shape::new([3]))?;

        assert_eq!(tensor.shape().dims(), &[3]);
        assert_eq!(tensor.numel(), 3);

        runtime.synchronize()?;

        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA-capable GPU"]
    fn allocate_zero_tensor() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;

        let tensor = runtime.zeros::<f32>(Shape::new([2, 4]))?;

        assert_eq!(tensor.shape().dims(), &[2, 4]);
        assert_eq!(tensor.numel(), 8);

        runtime.synchronize()?;

        Ok(())
    }
}
