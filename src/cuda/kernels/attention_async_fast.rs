use std::sync::Arc;

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::{attention::PagedAttentionLaunch, kernel_set::KernelSet};

const BLOCK_SIZE: u32 = 256;
pub(crate) const SPLITK_PARTIAL_STRIDE: usize = 66;

pub(crate) struct FastRaggedAttentionLaunch<'a> {
    pub(crate) page_size: usize,
    pub(crate) query: &'a CudaSlice<bf16>,
    pub(crate) key_cache: &'a CudaSlice<bf16>,
    pub(crate) value_cache: &'a CudaSlice<bf16>,
    pub(crate) block_tables: &'a CudaSlice<u32>,
    pub(crate) request_slots: &'a CudaSlice<u32>,
    pub(crate) position_ids: &'a CudaSlice<u32>,
    pub(crate) output: &'a mut CudaSlice<bf16>,
    pub(crate) num_tokens: usize,
    pub(crate) num_pages: usize,
    pub(crate) block_table_stride: usize,
}

pub(crate) struct SplitKRaggedAttentionLaunch<'a> {
    pub(crate) page_size: usize,
    pub(crate) query: &'a CudaSlice<bf16>,
    pub(crate) key_cache: &'a CudaSlice<bf16>,
    pub(crate) value_cache: &'a CudaSlice<bf16>,
    pub(crate) block_tables: &'a CudaSlice<u32>,
    pub(crate) request_slots: &'a CudaSlice<u32>,
    pub(crate) position_ids: &'a CudaSlice<u32>,
    pub(crate) partials: &'a mut CudaSlice<f32>,
    pub(crate) output: &'a mut CudaSlice<bf16>,
    pub(crate) num_tokens: usize,
    pub(crate) num_pages: usize,
    pub(crate) block_table_stride: usize,
    pub(crate) num_splits: usize,
}

pub(crate) struct AsyncAttentionFastKernels {
    ps16: KernelLaunch,
    ps32: KernelLaunch,
    ragged_ps16: KernelLaunch,
    ragged_ps32: KernelLaunch,
    splitk_ragged_ps16: KernelLaunch,
    splitk_ragged_ps32: KernelLaunch,
    splitk_merge: KernelLaunch,
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
        let splitk_ragged_ps16 = load_function(
            &module,
            Self::MODULE_NAME,
            "paged_ragged_gqa_lfm2_bf16_splitk_ps16",
        )?;
        let splitk_ragged_ps32 = load_function(
            &module,
            Self::MODULE_NAME,
            "paged_ragged_gqa_lfm2_bf16_splitk_ps32",
        )?;
        let splitk_merge = load_function(
            &module,
            Self::MODULE_NAME,
            "merge_ragged_gqa_lfm2_bf16_splitk",
        )?;
        for (name, function) in [
            ("ps16", &ps16),
            ("ps32", &ps32),
            ("ragged_ps16", &ragged_ps16),
            ("ragged_ps32", &ragged_ps32),
            ("splitk_ragged_ps16", &splitk_ragged_ps16),
            ("splitk_ragged_ps32", &splitk_ragged_ps32),
            ("splitk_merge", &splitk_merge),
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
            splitk_ragged_ps16: KernelLaunch::new_with_multiple(
                splitk_ragged_ps16,
                BLOCK_SIZE,
                32,
            )?,
            splitk_ragged_ps32: KernelLaunch::new_with_multiple(
                splitk_ragged_ps32,
                BLOCK_SIZE,
                32,
            )?,
            splitk_merge: KernelLaunch::new_with_multiple(splitk_merge, BLOCK_SIZE, 32)?,
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

    pub(crate) unsafe fn launch_lfm2_bf16(
        &self,
        stream: &CudaStream,
        launch: PagedAttentionLaunch<'_>,
    ) -> Result<()> {
        let PagedAttentionLaunch {
            page_size,
            query,
            key_cache,
            value_cache,
            block_table,
            position_ids,
            output,
            num_tokens,
            num_pages,
        } = launch;
        ensure!(num_tokens > 0, "fast-exp attention requires tokens");
        ensure!(num_pages > 0, "fast-exp attention requires cache pages");
        ensure!(
            !block_table.is_empty(),
            "fast-exp attention block table is empty"
        );
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
        ensure!(
            value_cache.len() >= cache_required,
            "fast-exp V cache too small"
        );
        ensure!(
            position_ids.len() >= num_tokens,
            "fast-exp positions too small"
        );
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
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }

    fn validate_ragged_common(
        page_size: usize,
        query: &CudaSlice<bf16>,
        key_cache: &CudaSlice<bf16>,
        value_cache: &CudaSlice<bf16>,
        block_tables: &CudaSlice<u32>,
        request_slots: &CudaSlice<u32>,
        position_ids: &CudaSlice<u32>,
        output: &CudaSlice<bf16>,
        num_tokens: usize,
        num_pages: usize,
        block_table_stride: usize,
    ) -> Result<()> {
        ensure!(num_tokens > 0, "fast-exp ragged attention requires tokens");
        ensure!(
            num_pages > 0,
            "fast-exp ragged attention requires cache pages"
        );
        ensure!(
            matches!(page_size, 16 | 32),
            "unsupported fast-exp ragged page size {page_size}"
        );
        ensure!(
            block_table_stride > 0,
            "fast-exp ragged block table stride must be positive"
        );
        ensure!(
            block_tables.len() >= block_table_stride,
            "fast-exp ragged block tables too small"
        );
        ensure!(
            block_tables.len().is_multiple_of(block_table_stride),
            "fast-exp ragged block tables not row aligned"
        );
        ensure!(
            request_slots.len() >= num_tokens,
            "fast-exp ragged request slots too small"
        );
        ensure!(
            position_ids.len() >= num_tokens,
            "fast-exp ragged positions too small"
        );
        let q_required = num_tokens
            .checked_mul(32 * 64)
            .context("fast-exp ragged query size overflow")?;
        let cache_required = num_pages
            .checked_mul(8)
            .and_then(|value| value.checked_mul(page_size))
            .and_then(|value| value.checked_mul(64))
            .context("fast-exp ragged cache size overflow")?;
        ensure!(
            query.len() >= q_required,
            "fast-exp ragged query storage too small"
        );
        ensure!(
            output.len() >= q_required,
            "fast-exp ragged output storage too small"
        );
        ensure!(
            key_cache.len() >= cache_required,
            "fast-exp ragged K cache too small"
        );
        ensure!(
            value_cache.len() >= cache_required,
            "fast-exp ragged V cache too small"
        );
        Ok(())
    }

    pub(crate) unsafe fn launch_ragged_lfm2_bf16(
        &self,
        stream: &CudaStream,
        launch: FastRaggedAttentionLaunch<'_>,
    ) -> Result<()> {
        let FastRaggedAttentionLaunch {
            page_size,
            query,
            key_cache,
            value_cache,
            block_tables,
            request_slots,
            position_ids,
            output,
            num_tokens,
            num_pages,
            block_table_stride,
        } = launch;
        Self::validate_ragged_common(
            page_size,
            query,
            key_cache,
            value_cache,
            block_tables,
            request_slots,
            position_ids,
            output,
            num_tokens,
            num_pages,
            block_table_stride,
        )?;
        let kernel = match page_size {
            16 => &self.ragged_ps16,
            32 => &self.ragged_ps32,
            _ => unreachable!(),
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
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }

    pub(crate) unsafe fn launch_splitk_ragged_lfm2_bf16(
        &self,
        stream: &CudaStream,
        launch: SplitKRaggedAttentionLaunch<'_>,
    ) -> Result<()> {
        let SplitKRaggedAttentionLaunch {
            page_size,
            query,
            key_cache,
            value_cache,
            block_tables,
            request_slots,
            position_ids,
            partials,
            output,
            num_tokens,
            num_pages,
            block_table_stride,
            num_splits,
        } = launch;
        Self::validate_ragged_common(
            page_size,
            query,
            key_cache,
            value_cache,
            block_tables,
            request_slots,
            position_ids,
            output,
            num_tokens,
            num_pages,
            block_table_stride,
        )?;
        ensure!(
            (2..=8).contains(&num_splits),
            "split-K ragged attention requires 2..=8 splits"
        );
        let partial_required = num_tokens
            .checked_mul(32)
            .and_then(|value| value.checked_mul(num_splits))
            .and_then(|value| value.checked_mul(SPLITK_PARTIAL_STRIDE))
            .context("split-K partial workspace size overflow")?;
        ensure!(
            partials.len() >= partial_required,
            "split-K partial workspace too small: need {partial_required}, have {}",
            partials.len()
        );

        let split_kernel = match page_size {
            16 => &self.splitk_ragged_ps16,
            32 => &self.splitk_ragged_ps32,
            _ => unreachable!(),
        };
        let block_table_rows = block_tables.len() / block_table_stride;
        let num_splits_u32 = u32::try_from(num_splits).context("split count exceeds u32")?;
        let split_blocks = num_tokens
            .checked_mul(8)
            .and_then(|value| value.checked_mul(num_splits))
            .context("split-K grid size overflow")?;
        let split_config = Self::launch_config(split_blocks)?;
        {
            let mut split_args = stream.launch_builder(split_kernel.function());
            split_args
                .arg(query)
                .arg(key_cache)
                .arg(value_cache)
                .arg(block_tables)
                .arg(request_slots)
                .arg(position_ids)
                .arg(partials)
                .arg(&num_tokens)
                .arg(&num_pages)
                .arg(&block_table_stride)
                .arg(&block_table_rows)
                .arg(&num_splits_u32);
            unsafe {
                split_args.launch(split_config)?;
            }
        }

        let merge_blocks = num_tokens
            .checked_mul(8)
            .context("split-K merge grid size overflow")?;
        let merge_config = Self::launch_config(merge_blocks)?;
        let mut merge_args = stream.launch_builder(self.splitk_merge.function());
        merge_args
            .arg(&*partials)
            .arg(output)
            .arg(&num_tokens)
            .arg(&num_splits_u32);
        unsafe {
            merge_args.launch(merge_config)?;
        }
        Ok(())
    }
}
