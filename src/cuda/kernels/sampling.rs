use std::sync::Arc;

use anyhow::{Result, ensure};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const MAX_BLOCK_SIZE: u32 = 256;
const ARGMAX_LOGICAL_LANES: usize = 256;
const ARGMAX_STAGE1_BLOCKS_PER_ROW: usize = 32;
const ARGMAX_ATOMIC_BLOCKS_PER_ROW: usize = 32;
const ARGMAX_ATOMIC_MAX_COLUMNS: usize = 65_536;

pub(crate) struct SamplingKernels {
    argmax: KernelLaunch,
    argmax_rows: KernelLaunch,
    argmax_rows_stage1: KernelLaunch,
    argmax_rows_stage2: KernelLaunch,
    argmax_rows_atomic_stage1: KernelLaunch,
    argmax_rows_atomic_decode: KernelLaunch,
}

impl KernelSet for SamplingKernels {
    const MODULE_NAME: &'static str = "sampling";
    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/sampling.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let argmax = load_function(&module, Self::MODULE_NAME, "argmax_bf16")?;
        let argmax_rows = load_function(&module, Self::MODULE_NAME, "argmax_rows_bf16")?;
        let argmax_rows_stage1 =
            load_function(&module, Self::MODULE_NAME, "argmax_rows_bf16_stage1")?;
        let argmax_rows_stage2 =
            load_function(&module, Self::MODULE_NAME, "argmax_rows_bf16_stage2")?;
        let argmax_rows_atomic_stage1 =
            load_function(&module, Self::MODULE_NAME, "argmax_rows_bf16_atomic_stage1")?;
        let argmax_rows_atomic_decode =
            load_function(&module, Self::MODULE_NAME, "argmax_rows_bf16_atomic_decode")?;
        Ok(Self {
            argmax: KernelLaunch::new(argmax, MAX_BLOCK_SIZE)?,
            argmax_rows: KernelLaunch::new(argmax_rows, MAX_BLOCK_SIZE)?,
            argmax_rows_stage1: KernelLaunch::new(argmax_rows_stage1, MAX_BLOCK_SIZE)?,
            argmax_rows_stage2: KernelLaunch::new(argmax_rows_stage2, MAX_BLOCK_SIZE)?,
            argmax_rows_atomic_stage1: KernelLaunch::new(
                argmax_rows_atomic_stage1,
                MAX_BLOCK_SIZE,
            )?,
            argmax_rows_atomic_decode: KernelLaunch::new(
                argmax_rows_atomic_decode,
                MAX_BLOCK_SIZE,
            )?,
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

    pub(crate) unsafe fn launch_argmax_rows_bf16_multiblock(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<bf16>,
        partial_values: &mut CudaSlice<f32>,
        partial_indices: &mut CudaSlice<u32>,
        output: &mut CudaSlice<u32>,
        rows: usize,
        columns: usize,
    ) -> Result<()> {
        ensure!(rows > 0 && columns > 0, "multi-block argmax shape is empty");
        ensure!(
            columns <= u32::MAX as usize,
            "multi-block argmax columns exceed u32 index range"
        );
        let required = rows
            .checked_mul(columns)
            .ok_or_else(|| anyhow::anyhow!("multi-block argmax input size overflow"))?;
        let partial_elements = rows
            .checked_mul(ARGMAX_LOGICAL_LANES)
            .ok_or_else(|| anyhow::anyhow!("multi-block argmax scratch size overflow"))?;
        ensure!(input.len() >= required, "multi-block argmax input too small");
        ensure!(
            partial_values.len() >= partial_elements,
            "multi-block argmax value scratch too small"
        );
        ensure!(
            partial_indices.len() >= partial_elements,
            "multi-block argmax index scratch too small"
        );
        ensure!(output.len() >= rows, "multi-block argmax output too small");

        let stage1_blocks = rows
            .checked_mul(ARGMAX_STAGE1_BLOCKS_PER_ROW)
            .ok_or_else(|| anyhow::anyhow!("multi-block argmax grid size overflow"))?;
        let stage1_config = self.argmax_rows_stage1.policy().exact_blocks(stage1_blocks)?;
        let mut stage1 = stream.launch_builder(self.argmax_rows_stage1.function());
        stage1
            .arg(input)
            .arg(&mut *partial_values)
            .arg(&mut *partial_indices)
            .arg(&rows)
            .arg(&columns);
        unsafe {
            stage1.launch(stage1_config)?;
        }

        let stage2_config = self.argmax_rows_stage2.policy().exact_blocks(rows)?;
        let mut stage2 = stream.launch_builder(self.argmax_rows_stage2.function());
        stage2
            .arg(&*partial_values)
            .arg(&*partial_indices)
            .arg(output)
            .arg(&rows);
        unsafe {
            stage2.launch(stage2_config)?;
        }
        Ok(())
    }

    pub(crate) unsafe fn launch_argmax_rows_bf16_atomic(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<bf16>,
        output: &mut CudaSlice<u32>,
        rows: usize,
        columns: usize,
    ) -> Result<()> {
        ensure!(rows > 0 && columns > 0, "atomic argmax shape is empty");
        ensure!(
            columns <= ARGMAX_ATOMIC_MAX_COLUMNS,
            "atomic argmax requires columns <= {ARGMAX_ATOMIC_MAX_COLUMNS}, got {columns}"
        );
        let required = rows
            .checked_mul(columns)
            .ok_or_else(|| anyhow::anyhow!("atomic argmax input size overflow"))?;
        ensure!(input.len() >= required, "atomic argmax input too small");
        ensure!(output.len() >= rows, "atomic argmax output too small");

        stream.memset_zeros(&mut *output)?;

        let stage1_blocks = rows
            .checked_mul(ARGMAX_ATOMIC_BLOCKS_PER_ROW)
            .ok_or_else(|| anyhow::anyhow!("atomic argmax grid size overflow"))?;
        let stage1_config = self
            .argmax_rows_atomic_stage1
            .policy()
            .exact_blocks(stage1_blocks)?;
        let mut stage1 = stream.launch_builder(self.argmax_rows_atomic_stage1.function());
        stage1
            .arg(input)
            .arg(&mut *output)
            .arg(&rows)
            .arg(&columns);
        unsafe {
            stage1.launch(stage1_config)?;
        }

        let stage2_blocks = rows.div_ceil(MAX_BLOCK_SIZE as usize);
        let stage2_config = self
            .argmax_rows_atomic_decode
            .policy()
            .exact_blocks(stage2_blocks)?;
        let mut stage2 = stream.launch_builder(self.argmax_rows_atomic_decode.function());
        stage2.arg(&mut *output).arg(&rows);
        unsafe {
            stage2.launch(stage2_config)?;
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
