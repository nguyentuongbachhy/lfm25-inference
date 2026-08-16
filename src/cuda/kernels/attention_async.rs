use std::sync::Arc;

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const MAX_BLOCK_SIZE: u32 = 256;

pub(crate) struct AsyncAttentionKernels {
    ps16: KernelLaunch,
    ps32: KernelLaunch,
}

impl KernelSet for AsyncAttentionKernels {
    const MODULE_NAME: &'static str = "attention_async";
    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/attention_async.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let ps16 = load_function(
            &module,
            Self::MODULE_NAME,
            "paged_gqa_lfm2_bf16_async_ps16",
        )?;
        let ps32 = load_function(
            &module,
            Self::MODULE_NAME,
            "paged_gqa_lfm2_bf16_async_ps32",
        )?;

        Ok(Self {
            ps16: KernelLaunch::new_with_multiple(ps16, MAX_BLOCK_SIZE, 32)?,
            ps32: KernelLaunch::new_with_multiple(ps32, MAX_BLOCK_SIZE, 32)?,
        })
    }
}

impl AsyncAttentionKernels {
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
        ensure!(num_tokens > 0, "attention requires at least one token");
        ensure!(num_pages > 0, "attention requires at least one cache page");
        ensure!(!block_table.is_empty(), "attention block table must not be empty");

        let q_required = num_tokens
            .checked_mul(32 * 64)
            .context("attention query size overflow")?;
        let cache_required = num_pages
            .checked_mul(8)
            .and_then(|value| value.checked_mul(page_size))
            .and_then(|value| value.checked_mul(64))
            .context("attention cache size overflow")?;

        ensure!(query.len() >= q_required, "attention query storage too small");
        ensure!(output.len() >= q_required, "attention output storage too small");
        ensure!(
            key_cache.len() >= cache_required,
            "attention K cache storage too small"
        );
        ensure!(
            value_cache.len() >= cache_required,
            "attention V cache storage too small"
        );
        ensure!(
            position_ids.len() >= num_tokens,
            "attention position storage too small"
        );

        let kernel = match page_size {
            16 => &self.ps16,
            32 => &self.ps32,
            other => anyhow::bail!("unsupported attention page size {other}"),
        };
        let blocks = num_tokens
            .checked_mul(8)
            .context("attention grid size overflow")?;
        let config = kernel.policy().exact_blocks(blocks)?;
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

        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }
}
