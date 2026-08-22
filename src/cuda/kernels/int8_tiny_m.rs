use std::sync::Arc;

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const QUANT_BLOCK_SIZE: u32 = 256;
const LINEAR_BLOCK_SIZE: u32 = 128;
const WARPS_PER_LINEAR_BLOCK: usize = 4;
pub(crate) const INT8_TINY_M_LIMIT: usize = 8;
pub(crate) const W8A16_TINY_M_LIMIT: usize = 2;

pub(crate) struct QuantizeS8RowsLaunch<'a> {
    pub(crate) input: &'a CudaSlice<bf16>,
    pub(crate) output: &'a mut CudaSlice<i8>,
    pub(crate) scales: &'a mut CudaSlice<f32>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

pub(crate) struct TinyMInt8LinearLaunch<'a> {
    pub(crate) input: &'a CudaSlice<i8>,
    pub(crate) input_scales: &'a CudaSlice<f32>,
    pub(crate) weight: &'a CudaSlice<i8>,
    pub(crate) weight_scales: &'a CudaSlice<f32>,
    pub(crate) output: &'a mut CudaSlice<bf16>,
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
}

pub(crate) struct TinyMW8A16LinearLaunch<'a> {
    pub(crate) input: &'a CudaSlice<bf16>,
    pub(crate) weight: &'a CudaSlice<i8>,
    pub(crate) weight_scales: &'a CudaSlice<f32>,
    pub(crate) output: &'a mut CudaSlice<bf16>,
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
}

pub(crate) struct Int8TinyMKernels {
    quantize_rows: KernelLaunch,
    linear: KernelLaunch,
    weight_only_linear: KernelLaunch,
}

impl KernelSet for Int8TinyMKernels {
    const MODULE_NAME: &'static str = "int8_tiny_m";
    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/int8_tiny_m.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let quantize_rows = KernelLaunch::new_with_multiple(
            load_function(&module, Self::MODULE_NAME, "quantize_bf16_rows_s8")?,
            QUANT_BLOCK_SIZE,
            QUANT_BLOCK_SIZE,
        )?;
        let linear = KernelLaunch::new_with_multiple(
            load_function(&module, Self::MODULE_NAME, "int8_tiny_m_dp4a_bf16")?,
            LINEAR_BLOCK_SIZE,
            LINEAR_BLOCK_SIZE,
        )?;
        let weight_only_linear = KernelLaunch::new_with_multiple(
            load_function(&module, Self::MODULE_NAME, "int8_weight_bf16_tiny_m_bf16")?,
            LINEAR_BLOCK_SIZE,
            LINEAR_BLOCK_SIZE,
        )?;
        ensure!(
            quantize_rows.policy().block_size() == QUANT_BLOCK_SIZE,
            "INT8 row quantizer did not resolve to 256 threads"
        );
        ensure!(
            linear.policy().block_size() == LINEAR_BLOCK_SIZE,
            "INT8 tiny-M linear did not resolve to 128 threads"
        );
        ensure!(
            weight_only_linear.policy().block_size() == LINEAR_BLOCK_SIZE,
            "W8A16 tiny-M linear did not resolve to 128 threads"
        );
        Ok(Self {
            quantize_rows,
            linear,
            weight_only_linear,
        })
    }
}

impl Int8TinyMKernels {
    pub(crate) unsafe fn launch_quantize_rows(
        &self,
        stream: &CudaStream,
        launch: QuantizeS8RowsLaunch<'_>,
    ) -> Result<()> {
        let QuantizeS8RowsLaunch {
            input,
            output,
            scales,
            rows,
            cols,
        } = launch;
        ensure!(rows > 0 && cols > 0, "INT8 row quantization requires non-empty input");
        let elements = rows
            .checked_mul(cols)
            .context("INT8 row quantization size overflow")?;
        ensure!(input.len() >= elements, "INT8 row quantization input too small");
        ensure!(output.len() >= elements, "INT8 row quantization output too small");
        ensure!(scales.len() >= rows, "INT8 row quantization scale buffer too small");

        let config = self.quantize_rows.policy().exact_blocks(rows)?;
        let mut args = stream.launch_builder(self.quantize_rows.function());
        args.arg(input)
            .arg(output)
            .arg(scales)
            .arg(&rows)
            .arg(&cols);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }

    pub(crate) unsafe fn launch_linear(
        &self,
        stream: &CudaStream,
        launch: TinyMInt8LinearLaunch<'_>,
    ) -> Result<()> {
        let TinyMInt8LinearLaunch {
            input,
            input_scales,
            weight,
            weight_scales,
            output,
            m,
            n,
            k,
        } = launch;
        ensure!(
            (1..=INT8_TINY_M_LIMIT).contains(&m),
            "INT8 tiny-M linear supports M=1..={INT8_TINY_M_LIMIT}, got {m}"
        );
        ensure!(n > 0 && k > 0, "INT8 tiny-M linear requires non-empty N/K");
        ensure!(k.is_multiple_of(4), "INT8 tiny-M linear requires K divisible by 4");
        let input_elements = m.checked_mul(k).context("INT8 input size overflow")?;
        let weight_elements = n.checked_mul(k).context("INT8 weight size overflow")?;
        let output_elements = m.checked_mul(n).context("INT8 output size overflow")?;
        ensure!(input.len() >= input_elements, "INT8 tiny-M input buffer too small");
        ensure!(input_scales.len() >= m, "INT8 tiny-M input scales too small");
        ensure!(weight.len() >= weight_elements, "INT8 tiny-M weight buffer too small");
        ensure!(weight_scales.len() >= n, "INT8 tiny-M weight scales too small");
        ensure!(output.len() >= output_elements, "INT8 tiny-M output buffer too small");

        let blocks = n.div_ceil(WARPS_PER_LINEAR_BLOCK);
        let config = self.linear.policy().exact_blocks(blocks)?;
        let mut args = stream.launch_builder(self.linear.function());
        args.arg(input)
            .arg(input_scales)
            .arg(weight)
            .arg(weight_scales)
            .arg(output)
            .arg(&m)
            .arg(&n)
            .arg(&k);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }

    pub(crate) unsafe fn launch_weight_only_linear(
        &self,
        stream: &CudaStream,
        launch: TinyMW8A16LinearLaunch<'_>,
    ) -> Result<()> {
        let TinyMW8A16LinearLaunch {
            input,
            weight,
            weight_scales,
            output,
            m,
            n,
            k,
        } = launch;
        ensure!(
            (1..=W8A16_TINY_M_LIMIT).contains(&m),
            "W8A16 tiny-M linear supports M=1..={W8A16_TINY_M_LIMIT}, got {m}"
        );
        ensure!(n > 0 && k > 0, "W8A16 tiny-M linear requires non-empty N/K");
        ensure!(k.is_multiple_of(4), "W8A16 tiny-M linear requires K divisible by 4");
        let input_elements = m.checked_mul(k).context("W8A16 input size overflow")?;
        let weight_elements = n.checked_mul(k).context("W8A16 weight size overflow")?;
        let output_elements = m.checked_mul(n).context("W8A16 output size overflow")?;
        ensure!(input.len() >= input_elements, "W8A16 tiny-M input buffer too small");
        ensure!(weight.len() >= weight_elements, "W8A16 tiny-M weight buffer too small");
        ensure!(weight_scales.len() >= n, "W8A16 tiny-M weight scales too small");
        ensure!(output.len() >= output_elements, "W8A16 tiny-M output buffer too small");

        let blocks = n.div_ceil(WARPS_PER_LINEAR_BLOCK);
        let config = self.weight_only_linear.policy().exact_blocks(blocks)?;
        let mut args = stream.launch_builder(self.weight_only_linear.function());
        args.arg(input)
            .arg(weight)
            .arg(weight_scales)
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
