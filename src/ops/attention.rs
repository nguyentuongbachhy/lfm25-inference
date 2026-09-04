#[cfg(test)]
use anyhow::Context as _;
use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cache::PagedKvArena,
    cuda::{CudaRuntime, HybridAttentionLaunch},
    tensor::{Shape, Tensor},
};
#[cfg(test)]
use crate::{
    cache::PagedKvCache,
    cuda::{PagedAttentionLaunch, RaggedAttentionLaunch},
};

pub(crate) struct HybridRaggedAttentionInput<'a> {
    pub(crate) query: &'a Tensor<bf16>,
    pub(crate) current_key: &'a Tensor<bf16>,
    pub(crate) current_value: &'a Tensor<bf16>,
    pub(crate) arena: &'a PagedKvArena,
    pub(crate) block_tables: &'a Tensor<u32>,
    pub(crate) block_table_stride: usize,
    pub(crate) request_slots: &'a Tensor<u32>,
    pub(crate) position_ids: &'a Tensor<u32>,
    pub(crate) segment_offsets: &'a Tensor<u32>,
}

pub fn prefill_attention_lfm2_bf16(
    runtime: &CudaRuntime,
    query: &Tensor<bf16>,
    key: &Tensor<bf16>,
    value: &Tensor<bf16>,
) -> Result<Tensor<bf16>> {
    ensure!(
        query.rank() == 3 && query.dims()[1..] == [32, 64],
        "LFM2 query must have shape [N,32,64], got {:?}",
        query.dims()
    );
    let num_tokens = query.dims()[0];
    ensure!(
        key.dims() == [num_tokens, 8, 64],
        "LFM2 key must have shape [{num_tokens},8,64], got {:?}",
        key.dims()
    );
    ensure!(
        value.shape() == key.shape(),
        "LFM2 prefill K/V mismatch: K={:?}, V={:?}",
        key.dims(),
        value.dims()
    );
    let mut output = runtime.zeros::<bf16>(Shape::new([num_tokens, 32, 64]))?;
    unsafe {
        if super::prefill_dispatch::should_use_flash_prefill(num_tokens) {
            runtime
                .kernels()
                .attention()
                .launch_prefill_flash_lfm2_bf16(
                    runtime.stream(),
                    query.storage(),
                    key.storage(),
                    value.storage(),
                    output.storage_mut(),
                    num_tokens,
                )?;
        } else {
            runtime.kernels().attention().launch_prefill_lfm2_bf16(
                runtime.stream(),
                query.storage(),
                key.storage(),
                value.storage(),
                output.storage_mut(),
                num_tokens,
            )?;
        }
    }
    Ok(output)
}

pub fn segmented_prefill_attention_lfm2_bf16(
    runtime: &CudaRuntime,
    query: &Tensor<bf16>,
    key: &Tensor<bf16>,
    value: &Tensor<bf16>,
    segment_offsets: &Tensor<u32>,
    num_segments: usize,
    max_tokens_per_segment: usize,
) -> Result<Tensor<bf16>> {
    ensure!(
        query.rank() == 3 && query.dims()[1..] == [32, 64],
        "LFM2 query must have shape [N,32,64], got {:?}",
        query.dims()
    );
    let num_tokens = query.dims()[0];
    ensure!(
        key.dims() == [num_tokens, 8, 64],
        "LFM2 key must have shape [{num_tokens},8,64], got {:?}",
        key.dims()
    );
    ensure!(
        value.shape() == key.shape(),
        "LFM2 prefill K/V mismatch: K={:?}, V={:?}",
        key.dims(),
        value.dims()
    );
    ensure!(
        segment_offsets.numel() >= num_segments + 1,
        "segment offsets tensor too small"
    );
    let mut output = runtime.alloc_bf16(Shape::new([num_tokens, 32, 64]))?;
    unsafe {
        runtime
            .kernels()
            .attention()
            .launch_segmented_prefill_flash_lfm2_bf16(
                runtime.stream(),
                query.storage(),
                key.storage(),
                value.storage(),
                segment_offsets.storage(),
                output.storage_mut(),
                num_segments,
                max_tokens_per_segment,
                num_tokens,
            )?;
    }
    Ok(output)
}

#[cfg(test)]
pub(crate) fn paged_ragged_attention_lfm2_bf16(
    runtime: &CudaRuntime,
    query: &Tensor<bf16>,
    arena: &PagedKvArena,
    block_tables: &Tensor<u32>,
    block_table_stride: usize,
    request_slots: &Tensor<u32>,
    position_ids: &Tensor<u32>,
) -> Result<Tensor<bf16>> {
    ensure!(
        query.rank() == 3 && query.dims()[1..] == [32, 64],
        "ragged LFM2 query must have shape [N,32,64]"
    );
    ensure!(block_tables.rank() == 2, "block tables must have rank 2");
    ensure!(
        block_tables.dims()[1] == block_table_stride,
        "block table stride/shape mismatch"
    );
    let num_tokens = query.dims()[0];
    ensure!(
        request_slots.numel() == num_tokens,
        "request slot count mismatch"
    );
    ensure!(
        position_ids.numel() == num_tokens,
        "position count mismatch"
    );
    let maximum_position = block_table_stride
        .checked_mul(arena.page_size().value())
        .context("ragged attention capacity overflow")?;
    ensure!(maximum_position > 0, "ragged attention capacity is zero");
    let mut output = runtime.alloc_bf16(Shape::new([num_tokens, 32, 64]))?;
    unsafe {
        runtime.kernels().attention().launch_ragged_lfm2_bf16(
            runtime.stream(),
            RaggedAttentionLaunch {
                page_size: arena.page_size().value(),
                query: query.storage(),
                key_cache: arena.key().storage(),
                value_cache: arena.value().storage(),
                block_tables: block_tables.storage(),
                request_slots: request_slots.storage(),
                position_ids: position_ids.storage(),
                output: output.storage_mut(),
                num_tokens,
                num_pages: arena.num_pages(),
                block_table_length: block_table_stride,
                block_table_stride,
            },
        )?;
    }
    Ok(output)
}

pub(crate) fn hybrid_ragged_attention_lfm2_bf16(
    runtime: &CudaRuntime,
    input: HybridRaggedAttentionInput<'_>,
) -> Result<Tensor<bf16>> {
    let HybridRaggedAttentionInput {
        query,
        current_key,
        current_value,
        arena,
        block_tables,
        block_table_stride,
        request_slots,
        position_ids,
        segment_offsets,
    } = input;
    ensure!(
        query.rank() == 3 && query.dims()[1..] == [32, 64],
        "hybrid LFM2 query must have shape [N,32,64]"
    );
    let num_tokens = query.dims()[0];
    ensure!(
        current_key.dims() == [num_tokens, 8, 64] && current_value.shape() == current_key.shape(),
        "hybrid contiguous K/V shape mismatch"
    );
    ensure!(block_tables.rank() == 2, "block tables must have rank 2");
    ensure!(
        block_tables.dims()[1] == block_table_stride,
        "block table stride/shape mismatch"
    );
    ensure!(request_slots.numel() == num_tokens, "request slot mismatch");
    ensure!(position_ids.numel() == num_tokens, "position mismatch");
    ensure!(segment_offsets.numel() >= 2, "hybrid segments are empty");
    let num_segments = segment_offsets.numel() - 1;
    let mut output = runtime.alloc_bf16(Shape::new([num_tokens, 32, 64]))?;
    unsafe {
        runtime
            .kernels()
            .attention()
            .launch_hybrid_ragged_lfm2_bf16(
                runtime.stream(),
                HybridAttentionLaunch {
                    page_size: arena.page_size().value(),
                    query: query.storage(),
                    current_key: current_key.storage(),
                    current_value: current_value.storage(),
                    key_cache: arena.key().storage(),
                    value_cache: arena.value().storage(),
                    block_tables: block_tables.storage(),
                    request_slots: request_slots.storage(),
                    position_ids: position_ids.storage(),
                    segment_offsets: segment_offsets.storage(),
                    output: output.storage_mut(),
                    num_tokens,
                    num_pages: arena.num_pages(),
                    block_table_stride,
                    num_segments,
                },
            )?;
    }
    Ok(output)
}

#[cfg(test)]
pub(crate) fn paged_attention_lfm2_bf16_sync(
    runtime: &CudaRuntime,
    query: &Tensor<bf16>,
    cache: &PagedKvCache,
    position_ids: &Tensor<u32>,
) -> Result<Tensor<bf16>> {
    ensure!(
        query.rank() == 3 && query.dims()[1..] == [32, 64],
        "LFM2 query must have shape [N,32,64], got {:?}",
        query.dims()
    );
    let num_tokens = query.dims()[0];
    ensure!(
        position_ids.numel() == num_tokens,
        "attention position count mismatch: expected {num_tokens}, got {}",
        position_ids.numel()
    );
    let mut output = runtime.alloc_bf16(Shape::new([num_tokens, 32, 64]))?;
    unsafe {
        runtime.kernels().attention().launch_lfm2_bf16(
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
    Ok(output)
}
