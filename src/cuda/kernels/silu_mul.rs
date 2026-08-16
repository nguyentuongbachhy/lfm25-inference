use std::sync::Arc;

use anyhow::{Result, ensure};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const MAX_BLOCK_SIZE: u32 = 256;

const VECTOR_WIDTH: usize = 2;
#[allow(dead_code)]
const ITEMS_PER_THREAD: usize = 4;

pub(crate) struct SiluMulKernels {
    #[allow(dead_code)]
    silu_mul_bf16: KernelLaunch,
    silu_mul_packed_bf16: KernelLaunch,
}

impl KernelSet for SiluMulKernels {
    const MODULE_NAME: &'static str = "silu_mul";
    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/silu_mul.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let function = load_function(&module, Self::MODULE_NAME, "silu_mul_bf16")?;
        let packed_function = load_function(&module, Self::MODULE_NAME, "silu_mul_packed_bf16")?;

        let silu_mul_bf16 = KernelLaunch::new(function, MAX_BLOCK_SIZE)?;
        let silu_mul_packed_bf16 = KernelLaunch::new(packed_function, MAX_BLOCK_SIZE)?;

        Ok(Self {
            silu_mul_bf16,
            silu_mul_packed_bf16,
        })
    }
}

impl SiluMulKernels {
    #[allow(dead_code)]
    pub(crate) unsafe fn launch_bf16(
        &self,
        stream: &CudaStream,
        gate: &CudaSlice<bf16>,
        up: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        numel: usize,
    ) -> Result<()> {
        ensure!(numel > 0, "silu_mul requires at least one element",);

        ensure!(
            gate.len() >= numel,
            "silu_mul gate storage too small: \
             required={numel}, actual={}",
            gate.len(),
        );

        ensure!(
            up.len() >= numel,
            "silu_mul up storage too small: \
             required={numel}, actual={}",
            up.len(),
        );

        ensure!(
            out.len() >= numel,
            "silu_mul output storage too small: \
             required={numel}, actual={}",
            out.len(),
        );

        let vec_count = numel / VECTOR_WIDTH;

        let full_tiles = vec_count / ITEMS_PER_THREAD;

        let remainder = vec_count % ITEMS_PER_THREAD;

        let work_items = full_tiles + usize::from(remainder != 0);

        let work_items = work_items.max(1);

        let config = self.silu_mul_bf16.policy().for_work_items(work_items)?;

        let mut args = stream.launch_builder(self.silu_mul_bf16.function());

        args.arg(gate).arg(up).arg(out).arg(&numel);

        unsafe {
            args.launch(config)?;
        }

        Ok(())
    }

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

    #[cfg(test)]
    pub(crate) fn block_size(&self) -> u32 {
        self.silu_mul_bf16.policy().block_size()
    }
}
