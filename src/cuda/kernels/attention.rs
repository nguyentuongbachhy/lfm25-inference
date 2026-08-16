use std::sync::Arc;

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const MAX_BLOCK_SIZE: u32 = 256;

pub(crate) struct AttentionKernels {
    prefill: KernelLaunch,
    ps16: KernelLaunch,
    ps32: KernelLaunch,
    #[cfg(test)]
    ragged_ps16: KernelLaunch,
    #[cfg(test)]
    ragged_ps32: KernelLaunch,
    hybrid_ragged_ps16: KernelLaunch,
    hybrid_ragged_ps32: KernelLaunch,
}

impl KernelSet for AttentionKernels {
    const MODULE_NAME: &'static str = "attention";
    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/attention.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let prefill = load_function(&module, Self::MODULE_NAME, "prefill_gqa_lfm2_bf16")?;
        let ps16 = load_function(&module, Self::MODULE_NAME, "paged_gqa_lfm2_bf16_ps16")?;
        let ps32 = load_function(&module, Self::MODULE_NAME, "paged_gqa_lfm2_bf16_ps32")?;
        #[cfg(test)]
        let ragged_ps16 = load_function(
            &module,
            Self::MODULE_NAME,
            "paged_ragged_gqa_lfm2_bf16_ps16",
        )?;
        #[cfg(test)]
        let ragged_ps32 = load_function(
            &module,
            Self::MODULE_NAME,
            "paged_ragged_gqa_lfm2_bf16_ps32",
        )?;
        let hybrid_ragged_ps16 = load_function(
            &module,
            Self::MODULE_NAME,
            "hybrid_ragged_gqa_lfm2_bf16_ps16",
        )?;
        let hybrid_ragged_ps32 = load_function(
            &module,
            Self::MODULE_NAME,
            "hybrid_ragged_gqa_lfm2_bf16_ps32",
        )?;

        Ok(Self {
            prefill: KernelLaunch::new_with_multiple(prefill, MAX_BLOCK_SIZE, 32)?,
            ps16: KernelLaunch::new_with_multiple(ps16, MAX_BLOCK_SIZE, 32)?,
            ps32: KernelLaunch::new_with_multiple(ps32, MAX_BLOCK_SIZE, 32)?,
            #[cfg(test)]
            ragged_ps16: KernelLaunch::new_with_multiple(ragged_ps16, MAX_BLOCK_SIZE, 32)?,
            #[cfg(test)]
            ragged_ps32: KernelLaunch::new_with_multiple(ragged_ps32, MAX_BLOCK_SIZE, 32)?,
            hybrid_ragged_ps16: KernelLaunch::new_with_multiple(
                hybrid_ragged_ps16,
                MAX_BLOCK_SIZE,
                32,
            )?,
            hybrid_ragged_ps32: KernelLaunch::new_with_multiple(
                hybrid_ragged_ps32,
                MAX_BLOCK_SIZE,
                32,
            )?,
        })
    }
}

impl AttentionKernels {
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn launch_hybrid_ragged_lfm2_bf16(
        &self,
        stream: &CudaStream,
        page_size: usize,
        query: &CudaSlice<bf16>,
        current_key: &CudaSlice<bf16>,
        current_value: &CudaSlice<bf16>,
        key_cache: &CudaSlice<bf16>,
        value_cache: &CudaSlice<bf16>,
        block_tables: &CudaSlice<u32>,
        request_slots: &CudaSlice<u32>,
        position_ids: &CudaSlice<u32>,
        segment_offsets: &CudaSlice<u32>,
        output: &mut CudaSlice<bf16>,
        num_tokens: usize,
        num_pages: usize,
        block_table_stride: usize,
        num_segments: usize,
    ) -> Result<()> {
        ensure!(num_tokens > 0, "hybrid ragged attention requires tokens");
        ensure!(
            num_segments > 0,
            "hybrid ragged attention requires segments"
        );
        ensure!(
            num_pages > 0,
            "hybrid ragged attention requires cache pages"
        );
        ensure!(block_table_stride > 0, "hybrid block table is empty");
        ensure!(request_slots.len() >= num_tokens, "request slots too small");
        ensure!(position_ids.len() >= num_tokens, "positions too small");
        ensure!(
            segment_offsets.len() >= num_segments + 1,
            "segment offsets too small"
        );
        let q_required = num_tokens
            .checked_mul(32 * 64)
            .context("hybrid query size overflow")?;
        let kv_required = num_tokens
            .checked_mul(8 * 64)
            .context("hybrid current KV size overflow")?;
        let cache_required = num_pages
            .checked_mul(8)
            .and_then(|value| value.checked_mul(page_size))
            .and_then(|value| value.checked_mul(64))
            .context("hybrid cache size overflow")?;
        ensure!(
            query.len() >= q_required && output.len() >= q_required,
            "hybrid Q/O storage too small"
        );
        ensure!(
            current_key.len() >= kv_required && current_value.len() >= kv_required,
            "hybrid contiguous KV storage too small"
        );
        ensure!(
            key_cache.len() >= cache_required && value_cache.len() >= cache_required,
            "hybrid paged KV storage too small"
        );
        let kernel = match page_size {
            16 => &self.hybrid_ragged_ps16,
            32 => &self.hybrid_ragged_ps32,
            other => anyhow::bail!("unsupported hybrid attention page size {other}"),
        };
        let blocks = num_tokens.checked_mul(8).context("hybrid grid overflow")?;
        let config = kernel.policy().exact_blocks(blocks)?;
        let mut args = stream.launch_builder(kernel.function());
        args.arg(query)
            .arg(current_key)
            .arg(current_value)
            .arg(key_cache)
            .arg(value_cache)
            .arg(block_tables)
            .arg(request_slots)
            .arg(position_ids)
            .arg(segment_offsets)
            .arg(output)
            .arg(&num_tokens)
            .arg(&num_pages)
            .arg(&block_table_stride)
            .arg(&num_segments);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }

    #[cfg(test)]
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
        block_table_length: usize,
        block_table_stride: usize,
    ) -> Result<()> {
        ensure!(num_tokens > 0, "ragged attention requires tokens");
        ensure!(num_pages > 0, "ragged attention requires cache pages");
        ensure!(
            block_table_length > 0,
            "ragged block table must not be empty"
        );
        ensure!(
            block_table_stride >= block_table_length,
            "invalid block table stride"
        );
        ensure!(
            request_slots.len() >= num_tokens,
            "request slot storage too small"
        );
        ensure!(
            position_ids.len() >= num_tokens,
            "position storage too small"
        );
        let q_required = num_tokens
            .checked_mul(32 * 64)
            .context("ragged attention query size overflow")?;
        let cache_required = num_pages
            .checked_mul(8)
            .and_then(|value| value.checked_mul(page_size))
            .and_then(|value| value.checked_mul(64))
            .context("ragged attention cache size overflow")?;
        ensure!(
            query.len() >= q_required && output.len() >= q_required,
            "ragged Q/O storage too small"
        );
        ensure!(
            key_cache.len() >= cache_required && value_cache.len() >= cache_required,
            "ragged cache storage too small"
        );
        ensure!(
            block_tables.len() >= block_table_stride,
            "ragged block table storage too small"
        );

        let kernel = match page_size {
            16 => &self.ragged_ps16,
            32 => &self.ragged_ps32,
            other => anyhow::bail!("unsupported ragged attention page size {other}"),
        };
        let blocks = num_tokens.checked_mul(8).context("ragged grid overflow")?;
        let config = kernel.policy().exact_blocks(blocks)?;
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
            .arg(&block_table_length)
            .arg(&block_table_stride);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }

    pub(crate) unsafe fn launch_prefill_lfm2_bf16(
        &self,
        stream: &CudaStream,
        query: &CudaSlice<bf16>,
        key: &CudaSlice<bf16>,
        value: &CudaSlice<bf16>,
        output: &mut CudaSlice<bf16>,
        num_tokens: usize,
    ) -> Result<()> {
        ensure!(
            num_tokens > 0,
            "prefill attention requires at least one token"
        );
        let query_required = num_tokens
            .checked_mul(32 * 64)
            .context("prefill query size overflow")?;
        let kv_required = num_tokens
            .checked_mul(8 * 64)
            .context("prefill KV size overflow")?;
        ensure!(
            query.len() >= query_required,
            "prefill query storage too small"
        );
        ensure!(key.len() >= kv_required, "prefill key storage too small");
        ensure!(
            value.len() >= kv_required,
            "prefill value storage too small"
        );
        ensure!(
            output.len() >= query_required,
            "prefill output storage too small"
        );
        let query_tiles = num_tokens.div_ceil(2);
        let blocks = query_tiles
            .checked_mul(8)
            .context("prefill attention grid size overflow")?;
        let config = self.prefill.policy().exact_blocks(blocks)?;
        let mut args = stream.launch_builder(self.prefill.function());
        args.arg(query)
            .arg(key)
            .arg(value)
            .arg(output)
            .arg(&num_tokens);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }

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
        ensure!(
            !block_table.is_empty(),
            "attention block table must not be empty"
        );
        let q_required = num_tokens
            .checked_mul(32 * 64)
            .context("attention query size overflow")?;
        let cache_required = num_pages
            .checked_mul(8)
            .and_then(|value| value.checked_mul(page_size))
            .and_then(|value| value.checked_mul(64))
            .context("attention cache size overflow")?;

        ensure!(
            query.len() >= q_required,
            "attention query storage too small"
        );
        ensure!(
            output.len() >= q_required,
            "attention output storage too small"
        );
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
        let mut args = stream.launch_builder(kernel.function());
        let block_table_length = block_table.len();

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
