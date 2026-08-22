use std::sync::Arc;

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::{CudaModule, CudaStream, PushKernelArg};

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::{
    attention_async_fast::{
        FastRaggedAttentionLaunch, SPLITK_PARTIAL_STRIDE, SplitKRaggedAttentionLaunch,
    },
    kernel_set::KernelSet,
};

const BLOCK_SIZE: u32 = 128;

/// Test-only launcher for isolating CTA geometry from the production
/// attention algorithm. It reuses the exact same PTX/functions as the
/// 256-thread production path and changes only blockDim.x.
pub(crate) struct AttentionCta128Kernels {
    ragged_ps16: KernelLaunch,
    splitk_ragged_ps16: KernelLaunch,
    splitk_merge: KernelLaunch,
}

impl KernelSet for AttentionCta128Kernels {
    const MODULE_NAME: &'static str = "attention_async_fast_cta128_test";
    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/attention_async_fast.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let ragged_ps16 = KernelLaunch::new_with_multiple(
            load_function(
                &module,
                Self::MODULE_NAME,
                "paged_ragged_gqa_lfm2_bf16_async_fast_ps16",
            )?,
            BLOCK_SIZE,
            BLOCK_SIZE,
        )?;
        let splitk_ragged_ps16 = KernelLaunch::new_with_multiple(
            load_function(
                &module,
                Self::MODULE_NAME,
                "paged_ragged_gqa_lfm2_bf16_splitk_ps16",
            )?,
            BLOCK_SIZE,
            BLOCK_SIZE,
        )?;
        let splitk_merge = KernelLaunch::new_with_multiple(
            load_function(
                &module,
                Self::MODULE_NAME,
                "merge_ragged_gqa_lfm2_bf16_splitk",
            )?,
            BLOCK_SIZE,
            BLOCK_SIZE,
        )?;

        ensure!(
            ragged_ps16.policy().block_size() == BLOCK_SIZE
                && splitk_ragged_ps16.policy().block_size() == BLOCK_SIZE
                && splitk_merge.policy().block_size() == BLOCK_SIZE,
            "experimental attention launcher did not resolve to 128 threads"
        );

        Ok(Self {
            ragged_ps16,
            splitk_ragged_ps16,
            splitk_merge,
        })
    }
}

impl AttentionCta128Kernels {
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
        ensure!(page_size == 16, "CTA128 experiment currently supports PS16 only");
        ensure!(num_tokens > 0, "CTA128 attention requires tokens");
        ensure!(num_pages > 0, "CTA128 attention requires cache pages");
        ensure!(block_table_stride > 0, "CTA128 block-table stride must be positive");
        ensure!(
            block_tables.len().is_multiple_of(block_table_stride),
            "CTA128 block tables are not row aligned"
        );

        let blocks = num_tokens
            .checked_mul(8)
            .context("CTA128 attention grid size overflow")?;
        let config = self.ragged_ps16.policy().exact_blocks(blocks)?;
        let block_table_rows = block_tables.len() / block_table_stride;
        let mut args = stream.launch_builder(self.ragged_ps16.function());
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
        ensure!(page_size == 16, "CTA128 experiment currently supports PS16 only");
        ensure!((2..=8).contains(&num_splits), "CTA128 split-K requires 2..=8 splits");
        ensure!(
            block_tables.len().is_multiple_of(block_table_stride),
            "CTA128 block tables are not row aligned"
        );
        let partial_required = num_tokens
            .checked_mul(32)
            .and_then(|value| value.checked_mul(num_splits))
            .and_then(|value| value.checked_mul(SPLITK_PARTIAL_STRIDE))
            .context("CTA128 split-K partial size overflow")?;
        ensure!(partials.len() >= partial_required, "CTA128 split-K partial workspace too small");

        let block_table_rows = block_tables.len() / block_table_stride;
        let num_splits_u32 = u32::try_from(num_splits).context("split count exceeds u32")?;
        let split_blocks = num_tokens
            .checked_mul(8)
            .and_then(|value| value.checked_mul(num_splits))
            .context("CTA128 split-K grid size overflow")?;
        let split_config = self.splitk_ragged_ps16.policy().exact_blocks(split_blocks)?;
        {
            let mut args = stream.launch_builder(self.splitk_ragged_ps16.function());
            args.arg(query)
                .arg(key_cache)
                .arg(value_cache)
                .arg(block_tables)
                .arg(request_slots)
                .arg(position_ids)
                .arg(&mut *partials)
                .arg(&num_tokens)
                .arg(&num_pages)
                .arg(&block_table_stride)
                .arg(&block_table_rows)
                .arg(&num_splits_u32);
            unsafe { args.launch(split_config)?; }
        }

        let merge_blocks = num_tokens
            .checked_mul(8)
            .context("CTA128 merge grid size overflow")?;
        let merge_config = self.splitk_merge.policy().exact_blocks(merge_blocks)?;
        let mut merge_args = stream.launch_builder(self.splitk_merge.function());
        merge_args
            .arg(&*partials)
            .arg(output)
            .arg(&num_tokens)
            .arg(&num_splits_u32);
        unsafe { merge_args.launch(merge_config)?; }
        Ok(())
    }
}
