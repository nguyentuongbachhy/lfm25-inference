use std::sync::Arc;

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const BLOCK_SIZE: u32 = 256;

pub(crate) struct AsyncAttentionFastKernels {
    ps16: KernelLaunch,
    ps32: KernelLaunch,
    ragged_ps16: KernelLaunch,
    ragged_ps32: KernelLaunch,
}

impl KernelSet for AsyncAttentionFastKernels {
    const MODULE_NAME: &'static str = "attention_async_fast";
    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/attention_async_fast.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let ps16 = load_function(
            &module,
            Self::MODULE_NAME,
            "paged_gqa_lfm2_bf16_async_fast_ps16",
        )?;
        let ps32 = load_function(
            &module,
            Self::MODULE_NAME,
            "paged_gqa_lfm2_bf16_async_fast_ps32",
        )?;
        let ragged_ps16 = load_function(
            &module,
            Self::MODULE_NAME,
            "paged_ragged_gqa_lfm2_bf16_async_fast_ps16",
        )?;
        let ragged_ps32 = load_function(
            &module,
            Self::MODULE_NAME,
            "paged_ragged_gqa_lfm2_bf16_async_fast_ps32",
        )?;

        for (name, function) in [
            ("ps16", &ps16),
            ("ps32", &ps32),
            ("ragged_ps16", &ragged_ps16),
            ("ragged_ps32", &ragged_ps32),
        ] {
            ensure!(
                function.max_threads_per_block()? >= BLOCK_SIZE as i32,
                "fast-exp async attention {name} cannot launch required 256-thread block"
            );
        }

        Ok(Self {
            ps16: KernelLaunch::new_with_multiple(ps16, BLOCK_SIZE, 32)?,
            ps32: KernelLaunch::new_with_multiple(ps32, BLOCK_SIZE, 32)?,
            ragged_ps16: KernelLaunch::new_with_multiple(ragged_ps16, BLOCK_SIZE, 32)?,
            ragged_ps32: KernelLaunch::new_with_multiple(ragged_ps32, BLOCK_SIZE, 32)?,
        })
    }
}

impl AsyncAttentionFastKernels {
    fn launch_config(blocks: usize) -> Result<LaunchConfig> {
        let grid_x = u32::try_from(blocks).context("fast-exp attention grid size exceeds u32")?;
        Ok(LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn launch_lfm2_bf16(
        &self,
        stream: &CudaStream,
        page_size: usize,
        query: &CudaSlice<bf16>,
        key_cache: &CudaSlice<bf16>,
        value_cache: &CudaSlice<bf16>,
        block_table: &CudaSlice<u32>,
        position_ids: &CudaSlice<u32>,
        output: &mut CudaSlice<bf16>,
        num_tokens: usize,
        num_pages: usize,
    ) -> Result<()> {
        ensure!(num_tokens > 0, "fast-exp attention requires tokens");
        ensure!(num_pages > 0, "fast-exp attention requires cache pages");
        ensure!(!block_table.is_empty(), "fast-exp attention block table is empty");

        let q_required = num_tokens
            .checked_mul(32 * 64)
            .context("fast-exp attention query size overflow")?;
        let cache_required = num_pages
            .checked_mul(8)
            .and_then(|value| value.checked_mul(page_size))
            .and_then(|value| value.checked_mul(64))
            .context("fast-exp attention cache size overflow")?;
        ensure!(query.len() >= q_required, "fast-exp query storage too small");
        ensure!(output.len() >= q_required, "fast-exp output storage too small");
        ensure!(key_cache.len() >= cache_required, "fast-exp K cache too small");
        ensure!(value_cache.len() >= cache_required, "fast-exp V cache too small");
        ensure!(position_ids.len() >= num_tokens, "fast-exp positions too small");

        let kernel = match page_size {
            16 => &self.ps16,
            32 => &self.ps32,
            other => anyhow::bail!("unsupported fast-exp page size {other}"),
        };
        let blocks = num_tokens
            .checked_mul(8)
            .context("fast-exp attention grid size overflow")?;
        let config = Self::launch_config(blocks)?;
        let block_table_length = block_table.len();
        let mut args = stream.launch_builder(kernel.function());
        args.arg(query)
            .arg(key_cache)
            .arg(value_cache)
            .arg(block_table)
            .arg(position_ids)
            .arg(output)
            .arg(&num_tokens)
            .arg(&num_pages)
            .arg(&block_table_length);
        unsafe { args.launch(config)?; }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn launch_ragged_lfm2_bf16(
        &self,
        stream: &CudaStream,
        page_size: usize,
        query: &CudaSlice<bf16>,
        key_cache: &CudaSlice<bf16>,
        value_cache: &CudaSlice<bf16>,
        block_tables: &CudaSlice<u32>,
        request_slots: &CudaSlice<u32>,
        position_ids: &CudaSlice<u32>,
        output: &mut CudaSlice<bf16>,
        num_tokens: usize,
        num_pages: usize,
        block_table_stride: usize,
    ) -> Result<()> {
        ensure!(num_tokens > 0, "fast-exp ragged attention requires tokens");
        ensure!(num_pages > 0, "fast-exp ragged attention requires cache pages");
        ensure!(block_table_stride > 0, "fast-exp ragged block table stride must be positive");
        ensure!(block_tables.len() >= block_table_stride, "fast-exp ragged block tables too small");
        ensure!(block_tables.len() % block_table_stride == 0, "fast-exp ragged block tables not row aligned");
        ensure!(request_slots.len() >= num_tokens, "fast-exp ragged request slots too small");
        ensure!(position_ids.len() >= num_tokens, "fast-exp ragged positions too small");

        let q_required = num_tokens
            .checked_mul(32 * 64)
            .context("fast-exp ragged query size overflow")?;
        let cache_required = num_pages
            .checked_mul(8)
            .and_then(|value| value.checked_mul(page_size))
            .and_then(|value| value.checked_mul(64))
            .context("fast-exp ragged cache size overflow")?;
        ensure!(query.len() >= q_required, "fast-exp ragged query storage too small");
        ensure!(output.len() >= q_required, "fast-exp ragged output storage too small");
        ensure!(key_cache.len() >= cache_required, "fast-exp ragged K cache too small");
        ensure!(value_cache.len() >= cache_required, "fast-exp ragged V cache too small");

        let kernel = match page_size {
            16 => &self.ragged_ps16,
            32 => &self.ragged_ps32,
            other => anyhow::bail!("unsupported fast-exp ragged page size {other}"),
        };
        let blocks = num_tokens
            .checked_mul(8)
            .context("fast-exp ragged grid size overflow")?;
        let config = Self::launch_config(blocks)?;
        let block_table_rows = block_tables.len() / block_table_stride;
        let mut args = stream.launch_builder(kernel.function());
        args.arg(query)
            .arg(key_cache)
            .arg(value_cache)
            .arg(block_tables)
            .arg(request_slots)
            .arg(position_ids)
            .arg(output)
            .arg(&num_tokens)
            .arg(&num_pages)
            .arg(&block_table_stride)
            .arg(&block_table_rows);
        unsafe { args.launch(config)?; }
        Ok(())
    }
}
