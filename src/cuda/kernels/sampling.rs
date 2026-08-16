use std::sync::Arc;

use anyhow::{Result, ensure};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const MAX_BLOCK_SIZE: u32 = 256;

pub(crate) struct SamplingKernels {
    argmax: KernelLaunch,
    argmax_rows: KernelLaunch,
}

impl KernelSet for SamplingKernels {
    const MODULE_NAME: &'static str = "sampling";
    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/sampling.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let argmax = load_function(&module, Self::MODULE_NAME, "argmax_bf16")?;
        let argmax_rows = load_function(&module, Self::MODULE_NAME, "argmax_rows_bf16")?;
        Ok(Self {
            argmax: KernelLaunch::new(argmax, MAX_BLOCK_SIZE)?,
            argmax_rows: KernelLaunch::new(argmax_rows, MAX_BLOCK_SIZE)?,
        })
    }
}

impl SamplingKernels {
    pub(crate) unsafe fn launch_argmax_rows_bf16(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<bf16>,
        output: &mut CudaSlice<u32>,
        rows: usize,
        columns: usize,
    ) -> Result<()> {
        ensure!(rows > 0 && columns > 0, "batched argmax shape is empty");
        let required = rows
            .checked_mul(columns)
            .ok_or_else(|| anyhow::anyhow!("batched argmax size overflow"))?;
        ensure!(input.len() >= required, "batched argmax input too small");
        ensure!(output.len() >= rows, "batched argmax output too small");
        let config = self.argmax_rows.policy().exact_blocks(rows)?;
        let mut args = stream.launch_builder(self.argmax_rows.function());
        args.arg(input).arg(output).arg(&rows).arg(&columns);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }

    pub(crate) unsafe fn launch_argmax_bf16(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<bf16>,
        output: &mut CudaSlice<u32>,
        numel: usize,
    ) -> Result<()> {
        ensure!(numel > 0, "argmax requires a non-empty input");
        ensure!(
            numel <= u32::MAX as usize,
            "argmax input exceeds u32 index range"
        );
        ensure!(input.len() >= numel, "argmax input storage too small");
        ensure!(!output.is_empty(), "argmax output storage is empty");
        let config = self.argmax.policy().exact_blocks(1)?;
        let mut args = stream.launch_builder(self.argmax.function());
        args.arg(input).arg(output).arg(&numel);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }
}
