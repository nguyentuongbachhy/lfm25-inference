#![allow(dead_code)]

use std::sync::Arc;

use anyhow::Result;
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const MAX_BLOCK_SIZE: u32 = 256;
const PAIRS_PER_THREAD: usize = 4;

pub(crate) struct ResidualKernels {
    add_bf16: KernelLaunch,
}

impl KernelSet for ResidualKernels {
    const MODULE_NAME: &'static str = "residual";
    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/residual.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let function = load_function(&module, Self::MODULE_NAME, "residual_add_bf16")?;
        Ok(Self {
            add_bf16: KernelLaunch::new(function, MAX_BLOCK_SIZE)?,
        })
    }
}

impl ResidualKernels {
    pub(crate) unsafe fn launch_add_bf16(
        &self,
        stream: &CudaStream,
        residual: &CudaSlice<bf16>,
        update: &CudaSlice<bf16>,
        output: &mut CudaSlice<bf16>,
        numel: usize,
    ) -> Result<()> {
        let work_items = (numel / 2).div_ceil(PAIRS_PER_THREAD).max(1);
        let config = self.add_bf16.policy().for_work_items(work_items)?;
        let mut args = stream.launch_builder(self.add_bf16.function());
        args.arg(residual).arg(update).arg(output).arg(&numel);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }
}
