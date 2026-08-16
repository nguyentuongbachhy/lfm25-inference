use std::sync::Arc;

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const MAX_BLOCK_SIZE: u32 = 256;

pub(crate) struct GatherKernels {
    rows_bf16: KernelLaunch,
}

impl KernelSet for GatherKernels {
    const MODULE_NAME: &'static str = "gather";
    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/gather.ptx"));
    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let function = load_function(&module, Self::MODULE_NAME, "gather_rows_bf16")?;
        Ok(Self {
            rows_bf16: KernelLaunch::new(function, MAX_BLOCK_SIZE)?,
        })
    }
}

impl GatherKernels {
    pub(crate) unsafe fn launch_rows_bf16(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<bf16>,
        row_indices: &CudaSlice<u32>,
        output: &mut CudaSlice<bf16>,
        output_rows: usize,
        input_rows: usize,
        columns: usize,
    ) -> Result<()> {
        ensure!(
            output_rows > 0 && input_rows > 0 && columns > 0,
            "invalid gather shape"
        );
        ensure!(
            row_indices.len() >= output_rows,
            "gather row indices too small"
        );
        let input_required = input_rows
            .checked_mul(columns)
            .context("gather input overflow")?;
        let output_required = output_rows
            .checked_mul(columns)
            .context("gather output overflow")?;
        ensure!(input.len() >= input_required, "gather input too small");
        ensure!(output.len() >= output_required, "gather output too small");
        let config = self.rows_bf16.policy().exact_blocks(output_rows)?;
        let mut args = stream.launch_builder(self.rows_bf16.function());
        args.arg(input)
            .arg(row_indices)
            .arg(output)
            .arg(&output_rows)
            .arg(&input_rows)
            .arg(&columns);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }
}
