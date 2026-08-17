use std::sync::Arc;

use anyhow::{Result, ensure};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const MAX_BLOCK_SIZE: u32 = 256;
const VECTOR_WIDTH: usize = 2;

pub(crate) struct SiluMulKernels {
    silu_mul_packed_bf16: KernelLaunch,
}

impl KernelSet for SiluMulKernels {
    const MODULE_NAME: &'static str = "silu_mul";
    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/silu_mul.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let function = load_function(&module, Self::MODULE_NAME, "silu_mul_packed_bf16")?;
        Ok(Self {
            silu_mul_packed_bf16: KernelLaunch::new(function, MAX_BLOCK_SIZE)?,
        })
    }
}

impl SiluMulKernels {
    pub(crate) unsafe fn launch_packed_bf16(
        &self,
        stream: &CudaStream,
        packed: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        rows: usize,
        intermediate_size: usize,
    ) -> Result<()> {
        ensure!(
            rows > 0 && intermediate_size > 0,
            "packed silu_mul requires non-empty dimensions"
        );
        let output_elements = rows
            .checked_mul(intermediate_size)
            .ok_or_else(|| anyhow::anyhow!("packed silu_mul output size overflow"))?;
        let packed_elements = output_elements
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("packed silu_mul input size overflow"))?;
        ensure!(
            packed.len() >= packed_elements,
            "packed silu_mul input storage too small"
        );
        ensure!(
            out.len() >= output_elements,
            "packed silu_mul output storage too small"
        );
        let work_items = output_elements.div_ceil(VECTOR_WIDTH).max(1);
        let config = self
            .silu_mul_packed_bf16
            .policy()
            .for_work_items(work_items)?;
        let mut args = stream.launch_builder(self.silu_mul_packed_bf16.function());
        args.arg(packed).arg(out).arg(&rows).arg(&intermediate_size);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }
}
