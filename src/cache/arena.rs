use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cuda::CudaRuntime,
    tensor::{Shape, Tensor},
};

use super::KvPageSize;

/// Physical KV storage shared by every live sequence on one attention layer.
/// Logical ownership lives in the allocator/block tables, never in this arena.
pub struct PagedKvArena {
    page_size: KvPageSize,
    num_pages: usize,
    key: Tensor<bf16>,
    value: Tensor<bf16>,
}

impl PagedKvArena {
    pub fn new(runtime: &CudaRuntime, num_pages: usize, page_size: KvPageSize) -> Result<Self> {
        ensure!(num_pages > 0, "KV arena requires at least one page");
        let shape = Shape::new([num_pages, 8, page_size.value(), 64]);
        Ok(Self {
            page_size,
            num_pages,
            key: runtime.zeros::<bf16>(shape.clone())?,
            value: runtime.zeros::<bf16>(shape)?,
        })
    }

    pub fn page_size(&self) -> KvPageSize {
        self.page_size
    }

    pub fn num_pages(&self) -> usize {
        self.num_pages
    }

    pub fn key(&self) -> &Tensor<bf16> {
        &self.key
    }

    pub fn value(&self) -> &Tensor<bf16> {
        &self.value
    }

    pub(crate) fn kv_mut(&mut self) -> (&mut Tensor<bf16>, &mut Tensor<bf16>) {
        (&mut self.key, &mut self.value)
    }

    pub fn write_lfm2(
        &mut self,
        runtime: &CudaRuntime,
        key: &Tensor<bf16>,
        value: &Tensor<bf16>,
        physical_slots: &Tensor<i64>,
    ) -> Result<()> {
        ensure!(
            key.rank() == 3 && key.dims()[1..] == [8, 64],
            "LFM2 K must have shape [N,8,64], got {:?}",
            key.dims()
        );
        ensure!(value.shape() == key.shape(), "LFM2 K/V shape mismatch");
        let num_tokens = key.dims()[0];
        ensure!(
            physical_slots.numel() == num_tokens,
            "physical slot count mismatch"
        );
        unsafe {
            runtime.kernels().kv_cache().launch_write_lfm2_bf16(
                runtime.stream(),
                self.page_size.value(),
                key.storage(),
                value.storage(),
                self.key.storage_mut(),
                self.value.storage_mut(),
                physical_slots.storage(),
                num_tokens,
                self.num_pages,
            )?;
        }
        Ok(())
    }
}
