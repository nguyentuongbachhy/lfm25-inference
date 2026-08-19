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
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u8>(),
            std::mem::size_of_val(values),
        )
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
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u8>(),
            std::mem::size_of_val(values),
        )
    };
    destination.extend_from_slice(bytes);
}

/// Fixed-capacity device metadata for a ragged model step.
pub struct GpuBatch {
    maximum_tokens: usize,
    block_table_stride: usize,
    block_table_elements: usize,
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
    block_patch_staging: Tensor<u32>,
    block_patch_host: Vec<u32>,
    block_patch_lookup: Vec<usize>,
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
        let block_table_elements = request_slots
            .checked_mul(block_table_stride)
            .context("block table capacity overflow")?;
        ensure!(
            u32::try_from(block_table_elements.saturating_sub(1)).is_ok(),
            "block table flat indices exceed u32"
        );
        let block_patch_words = block_table_elements
            .checked_mul(2)
            .context("block-table patch staging overflow")?;
        let metadata_capacity = packed_metadata_bytes(maximum_tokens, request_slots)?;
        Ok(Self {
            maximum_tokens,
            block_table_stride,
            block_table_elements,
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
            block_patch_staging: runtime.zeros::<u32>(Shape::new([block_patch_words]))?,
            block_patch_host: Vec::with_capacity(block_patch_words),
            block_patch_lookup: vec![usize::MAX; block_table_elements],
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
        ensure!(self.pending_tokens > 0, "GPU batch step metadata is missing");

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
            runtime.kernels().metadata().launch_scatter(
                runtime.stream(),
                self.metadata_staging.storage(),
                self.token_ids.storage_mut(),
                self.positions.storage_mut(),
                self.request_slots.storage_mut(),
                self.physical_slots.storage_mut(),
                self.segment_offsets.storage_mut(),
                self.segment_slots.storage_mut(),
                self.output_rows.storage_mut(),
                tokens,
                segments,
            )?;
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
        self.output_rows
            .set_logical_shape(Shape::new([segments]))?;

        self.transfers.h2d_bytes = self
            .transfers
            .h2d_bytes
            .saturating_add(packed_bytes as u64);
        self.transfers.h2d_calls = self.transfers.h2d_calls.saturating_add(1);
        self.pending_tokens = 0;
        Ok(())
    }

    fn flush_block_table_patches(&mut self, runtime: &CudaRuntime) -> Result<()> {
        if self.block_patch_host.is_empty() {
            return Ok(());
        }
        ensure!(
            self.block_patch_host.len().is_multiple_of(2),
            "block-table patch staging must contain index/value pairs"
        );
        let patch_count = self.block_patch_host.len() / 2;
        ensure!(
            self.block_patch_host.len() <= self.block_patch_staging.storage_capacity(),
            "block-table patch batch exceeds device staging capacity"
        );
        runtime.upload_prefix(&self.block_patch_host, &mut self.block_patch_staging)?;
        self.block_patch_staging
            .set_logical_shape(Shape::new([self.block_patch_host.len()]))?;
        unsafe {
            runtime.kernels().metadata().launch_block_table_patches(
                runtime.stream(),
                self.block_patch_staging.storage(),
                self.block_tables.storage_mut(),
                patch_count,
            )?;
        }
        self.transfers.h2d_bytes = self
            .transfers
            .h2d_bytes
            .saturating_add(std::mem::size_of_val(self.block_patch_host.as_slice()) as u64);
        self.transfers.h2d_calls = self.transfers.h2d_calls.saturating_add(1);

        for patch in self.block_patch_host.chunks_exact(2) {
            self.block_patch_lookup[patch[0] as usize] = usize::MAX;
        }
        self.block_patch_host.clear();
        Ok(())
    }

    pub fn update_step(
        &mut self,
        runtime: &CudaRuntime,
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

        // All page-table writes queued since the preceding model step are
        // committed as one transfer before this step's kernels consume them.
        self.flush_block_table_patches(runtime)?;

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
        _runtime: &CudaRuntime,
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

        for (offset, &entry) in entries.iter().enumerate() {
            let flat_index = start
                .checked_add(offset)
                .context("block table patch index overflow")?;
            ensure!(
                flat_index < self.block_table_elements,
                "block table patch index exceeds capacity"
            );
            let queued = self.block_patch_lookup[flat_index];
            if queued == usize::MAX {
                let patch_index = self.block_patch_host.len() / 2;
                self.block_patch_lookup[flat_index] = patch_index;
                self.block_patch_host.push(u32::try_from(flat_index)?);
                self.block_patch_host.push(entry);
            } else {
                self.block_patch_host[queued * 2 + 1] = entry;
            }
        }
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
