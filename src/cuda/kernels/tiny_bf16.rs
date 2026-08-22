use std::sync::Arc;

use anyhow::{Result, ensure};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

pub(crate) const TINY_BF16_MAX_M: usize = 8;
const TINY_BF16_BLOCK_THREADS: u32 = 256;
const TINY_BF16_WARPS_PER_BLOCK: usize = 8;

pub(crate) struct TinyBf16Kernels {
    nt_m8: KernelLaunch,
}

impl KernelSet for TinyBf16Kernels {
    const MODULE_NAME: &'static str = "tiny_bf16";
    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/tiny_bf16.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let nt_m8 = load_function(&module, Self::MODULE_NAME, "tiny_bf16_nt_m8")?;
        Ok(Self {
            nt_m8: KernelLaunch::new(nt_m8, TINY_BF16_BLOCK_THREADS)?,
        })
    }
}

impl TinyBf16Kernels {
    pub(crate) unsafe fn launch_nt_m8(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<bf16>,
        weight: &CudaSlice<bf16>,
        output: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        ensure!(m > 0 && m <= TINY_BF16_MAX_M, "tiny BF16 requires 1 <= M <= 8");
        ensure!(n > 0 && k > 0, "tiny BF16 shape is empty");
        ensure!(k % 2 == 0, "tiny BF16 requires even K for bfloat162 loads");
        ensure!(input.len() >= m * k, "tiny BF16 input storage too small");
        ensure!(weight.len() >= n * k, "tiny BF16 weight storage too small");
        ensure!(output.len() >= m * n, "tiny BF16 output storage too small");

        let blocks = n.div_ceil(TINY_BF16_WARPS_PER_BLOCK);
        let config = self.nt_m8.policy().exact_blocks(blocks)?;
        let mut args = stream.launch_builder(self.nt_m8.function());
        args.arg(input)
            .arg(weight)
            .arg(output)
            .arg(&m)
            .arg(&n)
            .arg(&k);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }
}
