use anyhow::{Context as _, Result, ensure};
use half::bf16;

use crate::{
    cache::{PagedKvArena, PagedKvCache},
    cuda::{
        CudaRuntime, FastRaggedAttentionLaunch, PagedAttentionLaunch, SplitKRaggedAttentionLaunch,
    },
    tensor::{Shape, Tensor},
};

pub(crate) const SPLITK_MAX_SPLITS: usize = 8;
const SPLITK_PARTIAL_STRIDE: usize = 66;
const SPLITK_TARGET_BLOCKS: usize = 64;
const SPLITK_MIN_PAGES_PER_SPLIT: usize = 8;
const SPLITK_MIN_CONTEXT_TOKENS: usize = 1024;

pub(crate) struct FastRaggedAttentionInput<'a> {
    pub(crate) query: &'a Tensor<bf16>,
    pub(crate) arena: &'a PagedKvArena,
    pub(crate) block_tables: &'a Tensor<u32>,
    pub(crate) block_table_stride: usize,
    pub(crate) request_slots: &'a Tensor<u32>,
    pub(crate) position_ids: &'a Tensor<u32>,
}

pub(crate) fn splitk_workspace_elements(maximum_tokens: usize) -> Result<usize> {
    maximum_tokens
        .checked_mul(32)
        .and_then(|value| value.checked_mul(SPLITK_MAX_SPLITS))
        .and_then(|value| value.checked_mul(SPLITK_PARTIAL_STRIDE))
        .context("split-K decode workspace size overflow")
}

/// Choose enough KV-axis splits to expose roughly 64 CTAs at low decode batch,
/// but never split a short context into tiny page ranges. For the 8-KV-head
/// LFM2 topology this yields B1->8, B2->4, B4->2 and B>=8->1 at sufficiently
/// long context. Returning one means the existing single-CTA-per-KV-head path.
pub(crate) fn splitk_decode_splits(
    num_tokens: usize,
    maximum_context_tokens: usize,
    page_size: usize,
) -> usize {
    if num_tokens == 0
        || maximum_context_tokens < SPLITK_MIN_CONTEXT_TOKENS
        || !matches!(page_size, 16 | 32)
    {
        return 1;
    }
    let base_blocks = num_tokens.saturating_mul(8).max(1);
    let occupancy_splits = SPLITK_TARGET_BLOCKS
        .div_ceil(base_blocks)
        .clamp(1, SPLITK_MAX_SPLITS);
    let context_pages = maximum_context_tokens.div_ceil(page_size);
    let page_splits = (context_pages / SPLITK_MIN_PAGES_PER_SPLIT)
        .clamp(1, SPLITK_MAX_SPLITS);
    occupancy_splits.min(page_splits).max(1)
}

pub(crate) fn paged_attention_fast_lfm2_bf16(
    runtime: &CudaRuntime,
    query: &Tensor<bf16>,
    cache: &PagedKvCache,
    position_ids: &Tensor<u32>,
) -> Result<Tensor<bf16>> {
    ensure!(
        query.rank() == 3 && query.dims()[1..] == [32, 64],
        "fast LFM2 query must have shape [N,32,64]"
    );
    let num_tokens = query.dims()[0];
    let mut output = runtime.alloc_bf16(Shape::new([num_tokens, 32, 64]))?;
    paged_attention_fast_lfm2_bf16_into(runtime, query, cache, position_ids, &mut output)?;
    Ok(output)
}

pub(crate) fn paged_attention_fast_lfm2_bf16_into(
    runtime: &CudaRuntime,
    query: &Tensor<bf16>,
    cache: &PagedKvCache,
    position_ids: &Tensor<u32>,
    output: &mut Tensor<bf16>,
) -> Result<()> {
    ensure!(
        query.rank() == 3 && query.dims()[1..] == [32, 64],
        "fast LFM2 query must have shape [N,32,64]"
    );
    let num_tokens = query.dims()[0];
    ensure!(
        position_ids.numel() == num_tokens,
        "fast attention position count mismatch"
    );
    output.set_logical_shape(Shape::new([num_tokens, 32, 64]))?;
    unsafe {
        runtime.kernels().attention_async_fast().launch_lfm2_bf16(
            runtime.stream(),
            PagedAttentionLaunch {
                page_size: cache.page_size().value(),
                query: query.storage(),
                key_cache: cache.key().storage(),
                value_cache: cache.value().storage(),
                block_table: cache.block_table().storage(),
                position_ids: position_ids.storage(),
                output: output.storage_mut(),
                num_tokens,
                num_pages: cache.num_pages(),
            },
        )?;
    }
    Ok(())
}

pub(crate) fn paged_ragged_attention_fast_lfm2_bf16(
    runtime: &CudaRuntime,
    input: FastRaggedAttentionInput<'_>,
) -> Result<Tensor<bf16>> {
    ensure!(
        input.query.rank() == 3 && input.query.dims()[1..] == [32, 64],
        "fast ragged LFM2 query must have shape [N,32,64]"
    );
    let num_tokens = input.query.dims()[0];
    let mut output = runtime.alloc_bf16(Shape::new([num_tokens, 32, 64]))?;
    paged_ragged_attention_fast_lfm2_bf16_into(runtime, input, &mut output)?;
    Ok(output)
}

fn validate_ragged_input(input: &FastRaggedAttentionInput<'_>) -> Result<usize> {
    ensure!(
        input.query.rank() == 3 && input.query.dims()[1..] == [32, 64],
        "fast ragged LFM2 query must have shape [N,32,64]"
    );
    ensure!(
        input.block_tables.rank() == 2,
        "fast block tables must be rank 2"
    );
    ensure!(
        input.block_tables.dims()[1] == input.block_table_stride,
        "fast block table stride/shape mismatch"
    );
    let num_tokens = input.query.dims()[0];
    ensure!(
        input.request_slots.numel() == num_tokens,
        "fast ragged request slot count mismatch"
    );
    ensure!(
        input.position_ids.numel() == num_tokens,
        "fast ragged position count mismatch"
    );
    Ok(num_tokens)
}

pub(crate) fn paged_ragged_attention_fast_lfm2_bf16_into(
    runtime: &CudaRuntime,
    input: FastRaggedAttentionInput<'_>,
    output: &mut Tensor<bf16>,
) -> Result<()> {
    let num_tokens = validate_ragged_input(&input)?;
    output.set_logical_shape(Shape::new([num_tokens, 32, 64]))?;
    unsafe {
        runtime
            .kernels()
            .attention_async_fast()
            .launch_ragged_lfm2_bf16(
                runtime.stream(),
                FastRaggedAttentionLaunch {
                    page_size: input.arena.page_size().value(),
                    query: input.query.storage(),
                    key_cache: input.arena.key().storage(),
                    value_cache: input.arena.value().storage(),
                    block_tables: input.block_tables.storage(),
                    request_slots: input.request_slots.storage(),
                    position_ids: input.position_ids.storage(),
                    output: output.storage_mut(),
                    num_tokens,
                    num_pages: input.arena.num_pages(),
                    block_table_stride: input.block_table_stride,
                },
            )?;
    }
    Ok(())
}

pub(crate) fn paged_ragged_attention_splitk_lfm2_bf16_into(
    runtime: &CudaRuntime,
    input: FastRaggedAttentionInput<'_>,
    partials: &mut Tensor<f32>,
    num_splits: usize,
    output: &mut Tensor<bf16>,
) -> Result<()> {
    ensure!(
        (2..=SPLITK_MAX_SPLITS).contains(&num_splits),
        "split-K attention requires 2..={SPLITK_MAX_SPLITS} splits"
    );
    let num_tokens = validate_ragged_input(&input)?;
    partials.set_logical_shape(Shape::new([
        num_tokens,
        32,
        num_splits,
        SPLITK_PARTIAL_STRIDE,
    ]))?;
    output.set_logical_shape(Shape::new([num_tokens, 32, 64]))?;
    unsafe {
        runtime
            .kernels()
            .attention_async_fast()
            .launch_splitk_ragged_lfm2_bf16(
                runtime.stream(),
                SplitKRaggedAttentionLaunch {
                    page_size: input.arena.page_size().value(),
                    query: input.query.storage(),
                    key_cache: input.arena.key().storage(),
                    value_cache: input.arena.value().storage(),
                    block_tables: input.block_tables.storage(),
                    request_slots: input.request_slots.storage(),
                    position_ids: input.position_ids.storage(),
                    partials: partials.storage_mut(),
                    output: output.storage_mut(),
                    num_tokens,
                    num_pages: input.arena.num_pages(),
                    block_table_stride: input.block_table_stride,
                    num_splits,
                },
            )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitk_dispatch_targets_low_batch_long_context_only() {
        assert_eq!(splitk_decode_splits(1, 512, 16), 1);
        assert_eq!(splitk_decode_splits(1, 1024, 16), 8);
        assert_eq!(splitk_decode_splits(2, 2048, 16), 4);
        assert_eq!(splitk_decode_splits(4, 2048, 16), 2);
        assert_eq!(splitk_decode_splits(8, 2048, 16), 1);
        assert_eq!(splitk_decode_splits(1, 1024, 32), 4);
        assert_eq!(splitk_decode_splits(1, 2048, 32), 8);
    }

    #[test]
    fn splitk_workspace_is_bounded_for_serving_capacity() -> Result<()> {
        assert_eq!(splitk_workspace_elements(64)?, 64 * 32 * 8 * 66);
        Ok(())
    }
}
