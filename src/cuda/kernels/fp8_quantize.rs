use std::sync::Arc;

use anyhow::{Result, ensure};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, CudaView, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const MAX_BLOCK_SIZE: u32 = 256;

pub(crate) struct Fp8QuantizeKernels {
    quantize_bf16_e4m3: KernelLaunch,
}

impl KernelSet for Fp8QuantizeKernels {
    const MODULE_NAME: &'static str = "fp8_quantize";
    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/fp8_quantize.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let function = load_function(&module, Self::MODULE_NAME, "quantize_bf16_e4m3")?;

        Ok(Self {
            quantize_bf16_e4m3: KernelLaunch::new(function, MAX_BLOCK_SIZE)?,
        })
    }
}

impl Fp8QuantizeKernels {
    pub(crate) unsafe fn launch_bf16_e4m3(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<bf16>,
        output: &mut CudaSlice<u8>,
        numel: usize,
        scale: f32,
    ) -> Result<()> {
        ensure!(numel > 0, "FP8 quantization requires at least one element");
        ensure!(
            input.len() >= numel,
            "FP8 quantization input storage too small"
        );
        ensure!(
            output.len() >= numel,
            "FP8 quantization output storage too small"
        );
        ensure!(
            scale.is_finite() && scale > 0.0,
            "invalid FP8 quantization scale"
        );

        let work_items = numel.div_ceil(2);
        let config = self
            .quantize_bf16_e4m3
            .policy()
            .for_work_items(work_items)?;
        let mut args = stream.launch_builder(self.quantize_bf16_e4m3.function());
        args.arg(input).arg(output).arg(&numel).arg(&scale);

        unsafe {
            args.launch(config)?;
        }

        Ok(())
    }

    pub(crate) unsafe fn launch_bf16_e4m3_view(
        &self,
        stream: &CudaStream,
        input: &CudaView<'_, bf16>,
        output: &mut CudaSlice<u8>,
        numel: usize,
        scale: f32,
    ) -> Result<()> {
        ensure!(numel > 0, "FP8 quantization requires at least one element");
        ensure!(input.len() >= numel, "FP8 input view is too small");
        ensure!(
            output.len() >= numel,
            "FP8 quantization output is too small"
        );
        ensure!(
            scale.is_finite() && scale > 0.0,
            "invalid FP8 quantization scale"
        );
        let work_items = numel.div_ceil(2);
        let config = self
            .quantize_bf16_e4m3
            .policy()
            .for_work_items(work_items)?;
        let mut args = stream.launch_builder(self.quantize_bf16_e4m3.function());
        args.arg(input).arg(output).arg(&numel).arg(&scale);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }
}
