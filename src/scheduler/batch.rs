use anyhow::{Context as _, Result, ensure};
use serde::Serialize;

use crate::{
    cuda::CudaRuntime,
    tensor::{Shape, Tensor},
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct TransferCounters {
    pub h2d_bytes: u64,
    pub h2d_calls: u64,
    pub d2h_bytes: u64,
    pub d2h_calls: u64,
    pub d2d_bytes: u64,
    pub d2d_calls: u64,
}

fn align_up_8(value: usize) -> Result<usize> {
    value
        .checked_add(7)
        .map(|value| value & !7usize)
        .context("metadata alignment overflow")
}

fn packed_metadata_bytes(tokens: usize, segments: usize) -> Result<usize> {
    let token_u32_bytes = tokens
        .checked_mul(std::mem::size_of::<u32>())
        .context("token metadata size overflow")?;
    let physical_offset = align_up_8(
        token_u32_bytes
            .checked_mul(3)
            .context("token metadata prefix overflow")?,
    )?;
    let physical_end = physical_offset
        .checked_add(
            tokens
                .checked_mul(std::mem::size_of::<i64>())
                .context("physical-slot metadata size overflow")?,
        )
        .context("physical-slot metadata end overflow")?;
    let segment_u32s = segments
        .checked_mul(3)
        .and_then(|value| value.checked_add(1))
        .context("segment metadata count overflow")?;
    physical_end
        .checked_add(
            segment_u32s
                .checked_mul(std::mem::size_of::<u32>())
                .context("segment metadata size overflow")?,
        )
        .context("packed metadata size overflow")
}

#[inline]
fn append_u32_bytes(destination: &mut Vec<u8>, values: &[u32]) {
    if values.is_empty() {
        return;
    }
    // SAFETY: u32 has no padding. This is a byte-for-byte staging copy into a
    // preallocated host slab consumed by the same native CUDA ABI.
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    destination.extend_from_slice(bytes);
}

#[inline]
fn append_i64_bytes(destination: &mut Vec<u8>, values: &[i64]) {
    if values.is_empty() {
        return;
    }
    // SAFETY: i64 has no padding and the slab is padded to an 8-byte boundary
    // before this section so the CUDA kernel may reinterpret it directly.
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    destination.extend_from_slice(bytes);
}

/// Fixed-capacity device metadata for a ragged model step.
pub struct GpuBatch {
    maximum_tokens: usize,
    block_table_stride: usize,
    max_context_tokens: usize,
    token_ids: Tensor<u32>,
    positions: Tensor<u32>,
    request_slots: Tensor<u32>,
    physical_slots: Tensor<i64>,
    segment_offsets: Tensor<u32>,
    segment_slots: Tensor<u32>,
    output_rows: Tensor<u32>,
    block_tables: Tensor<u32>,
    metadata_staging: Tensor<u8>,
    metadata_host: Vec<u8>,
    pending_tokens: usize,
    transfers: TransferCounters,
}

impl GpuBatch {
    pub fn new(
        runtime: &CudaRuntime,
        maximum_tokens: usize,
        request_slots: usize,
        block_table_stride: usize,
    ) -> Result<Self> {
        ensure!(
            maximum_tokens > 0,
            "GPU batch token capacity must be positive"
        );
        ensure!(
            request_slots > 0,
            "GPU batch request capacity must be positive"
        );
        ensure!(
            block_table_stride > 0,
            "block table stride must be positive"
        );
        request_slots
            .checked_mul(block_table_stride)
            .context("block table capacity overflow")?;
        let metadata_capacity = packed_metadata_bytes(maximum_tokens, request_slots)?;
        Ok(Self {
            maximum_tokens,
            block_table_stride,
            max_context_tokens: 0,
            token_ids: runtime.zeros::<u32>(Shape::new([maximum_tokens]))?,
            positions: runtime.zeros::<u32>(Shape::new([maximum_tokens]))?,
            request_slots: runtime.zeros::<u32>(Shape::new([maximum_tokens]))?,
            physical_slots: runtime.zeros::<i64>(Shape::new([maximum_tokens]))?,
            segment_offsets: runtime.zeros::<u32>(Shape::new([request_slots + 1]))?,
            segment_slots: runtime.zeros::<u32>(Shape::new([request_slots]))?,
            output_rows: runtime.zeros::<u32>(Shape::new([request_slots]))?,
            block_tables: runtime.zeros::<u32>(Shape::new([request_slots, block_table_stride]))?,
            metadata_staging: runtime.zeros::<u8>(Shape::new([metadata_capacity]))?,
            metadata_host: Vec::with_capacity(metadata_capacity),
            pending_tokens: 0,
            transfers: TransferCounters::default(),
        })
    }

    pub fn update_segments(
        &mut self,
        runtime: &CudaRuntime,
        segment_offsets: &[u32],
        segment_slots: &[u32],
        output_rows: &[u32],
    ) -> Result<()> {
        ensure!(!segment_slots.is_empty(), "GPU batch needs segments");
        ensure!(
            segment_offsets.len() == segment_slots.len() + 1,
            "segment offset count mismatch"
        );
        ensure!(
            output_rows.len() == segment_slots.len(),
            "output row count mismatch"
        );
        ensure!(
            self.pending_tokens > 0,
            "GPU batch step metadata is missing"
        );

        let tokens = self.pending_tokens;
        let segments = segment_slots.len();
        append_u32_bytes(&mut self.metadata_host, segment_offsets);
        append_u32_bytes(&mut self.metadata_host, segment_slots);
        append_u32_bytes(&mut self.metadata_host, output_rows);

        let packed_bytes = packed_metadata_bytes(tokens, segments)?;
        ensure!(
            self.metadata_host.len() == packed_bytes,
            "packed metadata layout mismatch: expected {packed_bytes} bytes, got {}",
            self.metadata_host.len()
        );
        ensure!(
            packed_bytes <= self.metadata_staging.storage_capacity(),
            "packed metadata exceeds staging capacity"
        );

        runtime.upload_prefix(&self.metadata_host, &mut self.metadata_staging)?;
        self.metadata_staging
            .set_logical_shape(Shape::new([packed_bytes]))?;

        unsafe {
            runtime
                .kernels()
                .metadata()
                .launch_scatter(crate::cuda::ScatterMetadataLaunch {
                    stream: runtime.stream(),
                    packed: self.metadata_staging.storage(),
                    token_ids: self.token_ids.storage_mut(),
                    positions: self.positions.storage_mut(),
                    request_slots: self.request_slots.storage_mut(),
                    physical_slots: self.physical_slots.storage_mut(),
                    segment_offsets: self.segment_offsets.storage_mut(),
                    segment_slots: self.segment_slots.storage_mut(),
                    output_rows: self.output_rows.storage_mut(),
                    num_tokens: tokens,
                    num_segments: segments,
                })?;
        }

        self.token_ids.set_logical_shape(Shape::new([tokens]))?;
        self.positions.set_logical_shape(Shape::new([tokens]))?;
        self.request_slots.set_logical_shape(Shape::new([tokens]))?;
        self.physical_slots
            .set_logical_shape(Shape::new([tokens]))?;
        self.segment_offsets
            .set_logical_shape(Shape::new([segment_offsets.len()]))?;
        self.segment_slots
            .set_logical_shape(Shape::new([segments]))?;
        self.output_rows.set_logical_shape(Shape::new([segments]))?;

        self.transfers.h2d_bytes = self.transfers.h2d_bytes.saturating_add(packed_bytes as u64);
        self.transfers.h2d_calls = self.transfers.h2d_calls.saturating_add(1);
        self.pending_tokens = 0;
        Ok(())
    }

    pub fn update_step(
        &mut self,
        _runtime: &CudaRuntime,
        token_ids: &[u32],
        positions: &[u32],
        request_slots: &[u32],
        physical_slots: &[i64],
    ) -> Result<()> {
        let tokens = token_ids.len();
        ensure!(
            tokens > 0 && tokens <= self.maximum_tokens,
            "invalid GPU batch size {tokens}"
        );
        ensure!(positions.len() == tokens, "position count mismatch");
        ensure!(request_slots.len() == tokens, "request slot count mismatch");
        ensure!(
            physical_slots.len() == tokens,
            "physical slot count mismatch"
        );
        ensure!(
            self.pending_tokens == 0,
            "previous GPU batch metadata was not committed"
        );

        let max_position = positions
            .iter()
            .copied()
            .max()
            .context("GPU batch positions are unexpectedly empty")?;
        self.max_context_tokens = usize::try_from(max_position)
            .context("GPU batch position exceeds usize")?
            .checked_add(1)
            .context("GPU batch context length overflow")?;

        self.metadata_host.clear();
        append_u32_bytes(&mut self.metadata_host, token_ids);
        append_u32_bytes(&mut self.metadata_host, positions);
        append_u32_bytes(&mut self.metadata_host, request_slots);
        let physical_offset = align_up_8(self.metadata_host.len())?;
        self.metadata_host.resize(physical_offset, 0);
        append_i64_bytes(&mut self.metadata_host, physical_slots);
        self.pending_tokens = tokens;
        Ok(())
    }

    pub fn update_block_table_range(
        &mut self,
        runtime: &CudaRuntime,
        request_slot: usize,
        logical_page_start: usize,
        entries: &[u32],
    ) -> Result<()> {
        ensure!(
            logical_page_start.saturating_add(entries.len()) <= self.block_table_stride,
            "block table update exceeds row"
        );
        let row_start = request_slot
            .checked_mul(self.block_table_stride)
            .context("block table offset overflow")?;
        let start = row_start
            .checked_add(logical_page_start)
            .context("block table update offset overflow")?;
        runtime.upload_range(entries, &mut self.block_tables, start)?;
        self.transfers.h2d_bytes = self
            .transfers
            .h2d_bytes
            .saturating_add(std::mem::size_of_val(entries) as u64);
        self.transfers.h2d_calls = self.transfers.h2d_calls.saturating_add(1);
        Ok(())
    }

    pub fn token_ids(&self) -> &Tensor<u32> {
        &self.token_ids
    }
    pub fn positions(&self) -> &Tensor<u32> {
        &self.positions
    }
    pub fn request_slots(&self) -> &Tensor<u32> {
        &self.request_slots
    }
    pub fn physical_slots(&self) -> &Tensor<i64> {
        &self.physical_slots
    }
    pub fn block_tables(&self) -> &Tensor<u32> {
        &self.block_tables
    }
    pub fn segment_offsets(&self) -> &Tensor<u32> {
        &self.segment_offsets
    }
    pub fn segment_slots(&self) -> &Tensor<u32> {
        &self.segment_slots
    }
    pub fn output_rows(&self) -> &Tensor<u32> {
        &self.output_rows
    }
    pub fn block_table_stride(&self) -> usize {
        self.block_table_stride
    }
    pub(crate) fn max_context_tokens(&self) -> usize {
        self.max_context_tokens
    }
    pub fn transfers(&self) -> TransferCounters {
        self.transfers
    }

    pub(crate) fn reset_transfers(&mut self) {
        self.transfers = TransferCounters::default();
    }
}
