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
        runtime.upload_prefix(segment_offsets, &mut self.segment_offsets)?;
        runtime.upload_prefix(segment_slots, &mut self.segment_slots)?;
        runtime.upload_prefix(output_rows, &mut self.output_rows)?;
        self.segment_offsets
            .set_logical_shape(Shape::new([segment_offsets.len()]))?;
        self.segment_slots
            .set_logical_shape(Shape::new([segment_slots.len()]))?;
        self.output_rows
            .set_logical_shape(Shape::new([output_rows.len()]))?;
        let bytes = std::mem::size_of_val(segment_offsets)
            .saturating_add(std::mem::size_of_val(segment_slots))
            .saturating_add(std::mem::size_of_val(output_rows));
        self.transfers.h2d_bytes = self.transfers.h2d_bytes.saturating_add(bytes as u64);
        self.transfers.h2d_calls = self.transfers.h2d_calls.saturating_add(3);
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

        let max_position = positions
            .iter()
            .copied()
            .max()
            .context("GPU batch positions are unexpectedly empty")?;
        self.max_context_tokens = usize::try_from(max_position)
            .context("GPU batch position exceeds usize")?
            .checked_add(1)
            .context("GPU batch context length overflow")?;

        runtime.upload_prefix(token_ids, &mut self.token_ids)?;
        runtime.upload_prefix(positions, &mut self.positions)?;
        runtime.upload_prefix(request_slots, &mut self.request_slots)?;
        runtime.upload_prefix(physical_slots, &mut self.physical_slots)?;
        self.token_ids.set_logical_shape(Shape::new([tokens]))?;
        self.positions.set_logical_shape(Shape::new([tokens]))?;
        self.request_slots.set_logical_shape(Shape::new([tokens]))?;
        self.physical_slots
            .set_logical_shape(Shape::new([tokens]))?;
        let bytes = tokens
            .checked_mul(std::mem::size_of::<u32>() * 3 + std::mem::size_of::<i64>())
            .context("transfer byte counter overflow")?;
        self.transfers.h2d_bytes = self.transfers.h2d_bytes.saturating_add(bytes as u64);
        self.transfers.h2d_calls = self.transfers.h2d_calls.saturating_add(4);
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
