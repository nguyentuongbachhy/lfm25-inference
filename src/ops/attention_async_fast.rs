use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cache::{PagedKvArena, PagedKvCache},
    cuda::{CudaRuntime, FastRaggedAttentionLaunch, PagedAttentionLaunch},
    tensor::{Shape, Tensor},
};

pub(crate) struct FastRaggedAttentionInput<'a> {
    pub(crate) query: &'a Tensor<bf16>,
    pub(crate) arena: &'a PagedKvArena,
    pub(crate) block_tables: &'a Tensor<u32>,
    pub(crate) block_table_stride: usize,
    pub(crate) request_slots: &'a Tensor<u32>,
    pub(crate) position_ids: &'a Tensor<u32>,
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
    ensure!(
        position_ids.numel() == num_tokens,
        "fast attention position count mismatch"
    );
    let mut output = runtime.alloc_bf16(Shape::new([num_tokens, 32, 64]))?;
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
    Ok(output)
}

pub(crate) fn paged_ragged_attention_fast_lfm2_bf16(
    runtime: &CudaRuntime,
    input: FastRaggedAttentionInput<'_>,
) -> Result<Tensor<bf16>> {
    let FastRaggedAttentionInput {
        query,
        arena,
        block_tables,
        block_table_stride,
        request_slots,
        position_ids,
    } = input;
    ensure!(
        query.rank() == 3 && query.dims()[1..] == [32, 64],
        "fast ragged LFM2 query must have shape [N,32,64]"
    );
    ensure!(block_tables.rank() == 2, "fast block tables must be rank 2");
    ensure!(
        block_tables.dims()[1] == block_table_stride,
        "fast block table stride/shape mismatch"
    );
    let num_tokens = query.dims()[0];
    ensure!(
        request_slots.numel() == num_tokens,
        "fast ragged request slot count mismatch"
    );
    ensure!(
        position_ids.numel() == num_tokens,
        "fast ragged position count mismatch"
    );

    let mut output = runtime.alloc_bf16(Shape::new([num_tokens, 32, 64]))?;
    unsafe {
        runtime
            .kernels()
            .attention_async_fast()
            .launch_ragged_lfm2_bf16(
                runtime.stream(),
                FastRaggedAttentionLaunch {
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
                    block_table_stride,
                },
            )?;
    }
    Ok(output)
}
