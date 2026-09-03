use std::sync::Arc;

use anyhow::{Result, ensure};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const MAX_BLOCK_SIZE: u32 = 256;
const WARP_SIZE: u32 = 32;

pub(crate) struct ResidualRmsNormFp8Launch<'a> {
    pub(crate) residual: &'a CudaSlice<bf16>,
    pub(crate) update: &'a CudaSlice<bf16>,
    pub(crate) weight: &'a CudaSlice<bf16>,
    pub(crate) residual_out: &'a mut CudaSlice<bf16>,
    pub(crate) normalized_fp8_out: &'a mut CudaSlice<u8>,
    pub(crate) rows: usize,
    pub(crate) hidden_size: usize,
    pub(crate) eps: f32,
    pub(crate) quant_scale: f32,
}

pub(crate) struct ResidualRmsFp8Kernels {
    residual_rms_norm_bf16_to_e4m3: KernelLaunch,
}

impl KernelSet for ResidualRmsFp8Kernels {
    const MODULE_NAME: &'static str = "residual_rms_fp8";
    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/residual_rms_fp8.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let function = load_function(&module, Self::MODULE_NAME, "residual_rms_norm_bf16_to_e4m3")?;
        let kernel = KernelLaunch::new_with_multiple(function, MAX_BLOCK_SIZE, WARP_SIZE)?;
        ensure!(
            kernel.policy().block_size() >= WARP_SIZE,
            "residual RMSNorm FP8 fusion requires at least one warp"
        );
        Ok(Self {
            residual_rms_norm_bf16_to_e4m3: kernel,
        })
    }
}

impl ResidualRmsFp8Kernels {
    pub(crate) unsafe fn launch(
        &self,
        stream: &CudaStream,
        launch: ResidualRmsNormFp8Launch<'_>,
    ) -> Result<()> {
        let ResidualRmsNormFp8Launch {
            residual,
            update,
            weight,
            residual_out,
            normalized_fp8_out,
            rows,
            hidden_size,
            eps,
            quant_scale,
        } = launch;
        ensure!(rows > 0, "residual RMSNorm FP8 fusion requires rows");
        ensure!(
            hidden_size > 0,
            "residual RMSNorm FP8 hidden size must be > 0"
        );
        ensure!(
            quant_scale.is_finite() && quant_scale > 0.0,
            "residual RMSNorm FP8 quantization scale must be finite and positive"
        );
        let config = self
            .residual_rms_norm_bf16_to_e4m3
            .policy()
            .exact_blocks(rows)?;
        let mut args = stream.launch_builder(self.residual_rms_norm_bf16_to_e4m3.function());
        args.arg(residual)
            .arg(update)
            .arg(weight)
            .arg(residual_out)
            .arg(normalized_fp8_out)
            .arg(&rows)
            .arg(&hidden_size)
            .arg(&eps)
            .arg(&quant_scale);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }
}
