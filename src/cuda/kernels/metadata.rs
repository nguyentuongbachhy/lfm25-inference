use std::sync::Arc;

use anyhow::{Result, ensure};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, PushKernelArg};

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const MAX_BLOCK_SIZE: u32 = 256;

pub(crate) struct MetadataKernels {
    scatter: KernelLaunch,
    block_table_patches: KernelLaunch,
}

impl KernelSet for MetadataKernels {
    const MODULE_NAME: &'static str = "metadata";
    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/metadata.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let scatter = load_function(&module, Self::MODULE_NAME, "scatter_batch_metadata")?;
        let block_table_patches =
            load_function(&module, Self::MODULE_NAME, "scatter_block_table_patches")?;
        Ok(Self {
            scatter: KernelLaunch::new(scatter, MAX_BLOCK_SIZE)?,
            block_table_patches: KernelLaunch::new(block_table_patches, MAX_BLOCK_SIZE)?,
        })
    }
}

impl MetadataKernels {
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn launch_scatter(
        &self,
        stream: &CudaStream,
        packed: &CudaSlice<u8>,
        token_ids: &mut CudaSlice<u32>,
        positions: &mut CudaSlice<u32>,
        request_slots: &mut CudaSlice<u32>,
        physical_slots: &mut CudaSlice<i64>,
        segment_offsets: &mut CudaSlice<u32>,
        segment_slots: &mut CudaSlice<u32>,
        output_rows: &mut CudaSlice<u32>,
        num_tokens: usize,
        num_segments: usize,
    ) -> Result<()> {
        ensure!(num_tokens > 0, "metadata scatter requires tokens");
        ensure!(num_segments > 0, "metadata scatter requires segments");
        ensure!(token_ids.len() >= num_tokens, "token metadata capacity is too small");
        ensure!(positions.len() >= num_tokens, "position metadata capacity is too small");
        ensure!(
            request_slots.len() >= num_tokens,
            "request-slot metadata capacity is too small"
        );
        ensure!(
            physical_slots.len() >= num_tokens,
            "physical-slot metadata capacity is too small"
        );
        ensure!(
            segment_offsets.len() > num_segments,
            "segment-offset metadata capacity is too small"
        );
        ensure!(
            segment_slots.len() >= num_segments && output_rows.len() >= num_segments,
            "segment metadata capacity is too small"
        );

        let work_items = num_tokens.max(num_segments + 1);
        let config = self.scatter.policy().for_work_items(work_items)?;
        let mut args = stream.launch_builder(self.scatter.function());
        args.arg(packed)
            .arg(token_ids)
            .arg(positions)
            .arg(request_slots)
            .arg(physical_slots)
            .arg(segment_offsets)
            .arg(segment_slots)
            .arg(output_rows)
            .arg(&num_tokens)
            .arg(&num_segments);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }

    pub(crate) unsafe fn launch_block_table_patches(
        &self,
        stream: &CudaStream,
        packed_pairs: &CudaSlice<u32>,
        block_tables: &mut CudaSlice<u32>,
        patch_count: usize,
    ) -> Result<()> {
        ensure!(patch_count > 0, "block-table scatter requires patches");
        ensure!(
            packed_pairs.len() >= patch_count.saturating_mul(2),
            "block-table patch staging capacity is too small"
        );
        let config = self
            .block_table_patches
            .policy()
            .for_work_items(patch_count)?;
        let mut args = stream.launch_builder(self.block_table_patches.function());
        args.arg(packed_pairs)
            .arg(block_tables)
            .arg(&patch_count);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }
}
