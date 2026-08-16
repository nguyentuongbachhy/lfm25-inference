use std::sync::Arc;

use anyhow::{Result, ensure};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const MAX_BLOCK_SIZE: u32 = 256;
const WARP_SIZE: u32 = 32;

pub(crate) struct RmsNormKernels {
    rms_norm_bf16: KernelLaunch,
    residual_rms_norm_bf16: KernelLaunch,
}

impl KernelSet for RmsNormKernels {
    const MODULE_NAME: &'static str = "rms_norm";

    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/rms_norm.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let function = load_function(&module, Self::MODULE_NAME, "rms_norm_bf16")?;
        let fused_function = load_function(&module, Self::MODULE_NAME, "residual_rms_norm_bf16")?;

        let rms_norm_bf16 = KernelLaunch::new_with_multiple(function, MAX_BLOCK_SIZE, WARP_SIZE)?;
        let residual_rms_norm_bf16 =
            KernelLaunch::new_with_multiple(fused_function, MAX_BLOCK_SIZE, WARP_SIZE)?;

        ensure!(
            rms_norm_bf16.policy().block_size() >= WARP_SIZE,
            "RMSNorm requires at least one warp",
        );

        ensure!(
            residual_rms_norm_bf16.policy().block_size() >= WARP_SIZE,
            "fused residual RMSNorm requires at least one warp",
        );

        Ok(Self {
            rms_norm_bf16,
            residual_rms_norm_bf16,
        })
    }
}

impl RmsNormKernels {
    pub(crate) unsafe fn launch_bf16(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<bf16>,
        weight: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        rows: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<()> {
        ensure!(rows > 0, "RMSNorm requires at least one row",);

        ensure!(hidden_size > 0, "RMSNorm hidden size must be > 0",);

        let config = self.rms_norm_bf16.policy().exact_blocks(rows)?;

        let mut args = stream.launch_builder(self.rms_norm_bf16.function());

        args.arg(x)
            .arg(weight)
            .arg(out)
            .arg(&rows)
            .arg(&hidden_size)
            .arg(&eps);

        unsafe {
            args.launch(config)?;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn launch_residual_bf16(
        &self,
        stream: &CudaStream,
        residual: &CudaSlice<bf16>,
        update: &CudaSlice<bf16>,
        weight: &CudaSlice<bf16>,
        residual_out: &mut CudaSlice<bf16>,
        normalized_out: &mut CudaSlice<bf16>,
        rows: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<()> {
        ensure!(rows > 0, "fused residual RMSNorm requires at least one row");
        ensure!(
            hidden_size > 0,
            "fused residual RMSNorm hidden size must be > 0"
        );
        let config = self.residual_rms_norm_bf16.policy().exact_blocks(rows)?;
        let mut args = stream.launch_builder(self.residual_rms_norm_bf16.function());
        args.arg(residual)
            .arg(update)
            .arg(weight)
            .arg(residual_out)
            .arg(normalized_out)
            .arg(&rows)
            .arg(&hidden_size)
            .arg(&eps);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }
}
