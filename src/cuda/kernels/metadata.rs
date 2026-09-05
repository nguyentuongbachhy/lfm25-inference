use std::sync::Arc;

use anyhow::{Result, ensure};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, PushKernelArg};

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const MAX_BLOCK_SIZE: u32 = 256;

pub(crate) struct MetadataKernels {
    scatter: KernelLaunch,
}

impl KernelSet for MetadataKernels {
    const MODULE_NAME: &'static str = "metadata";
    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/metadata.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let function = load_function(&module, Self::MODULE_NAME, "scatter_batch_metadata")?;
        Ok(Self {
            scatter: KernelLaunch::new(function, MAX_BLOCK_SIZE)?,
        })
    }
}

pub(crate) struct ScatterMetadataLaunch<'a> {
    pub(crate) stream: &'a CudaStream,
    pub(crate) packed: &'a CudaSlice<u8>,
    pub(crate) token_ids: &'a mut CudaSlice<u32>,
    pub(crate) positions: &'a mut CudaSlice<u32>,
    pub(crate) request_slots: &'a mut CudaSlice<u32>,
    pub(crate) physical_slots: &'a mut CudaSlice<i64>,
    pub(crate) segment_offsets: &'a mut CudaSlice<u32>,
    pub(crate) segment_slots: &'a mut CudaSlice<u32>,
    pub(crate) output_rows: &'a mut CudaSlice<u32>,
    pub(crate) num_tokens: usize,
    pub(crate) num_segments: usize,
}

impl MetadataKernels {
    pub(crate) unsafe fn launch_scatter(&self, launch: ScatterMetadataLaunch<'_>) -> Result<()> {
        let ScatterMetadataLaunch {
            stream,
            packed,
            token_ids,
            positions,
            request_slots,
            physical_slots,
            segment_offsets,
            segment_slots,
            output_rows,
            num_tokens,
            num_segments,
        } = launch;
        ensure!(num_tokens > 0, "metadata scatter requires tokens");
        ensure!(num_segments > 0, "metadata scatter requires segments");
        ensure!(
            token_ids.len() >= num_tokens,
            "token metadata capacity is too small"
        );
        ensure!(
            positions.len() >= num_tokens,
            "position metadata capacity is too small"
        );
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
}
