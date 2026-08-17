use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cache::{PagedKvArena, PagedKvCache},
    cuda::CudaRuntime,
    tensor::{Shape, Tensor},
};

pub fn paged_attention_lfm2_bf16(
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
        runtime.kernels().attention_async().launch_lfm2_bf16(
            runtime.stream(),
            cache.page_size().value(),
            query.storage(),
            cache.key().storage(),
            cache.value().storage(),
            cache.block_table().storage(),
            position_ids.storage(),
            output.storage_mut(),
            num_tokens,
            cache.num_pages(),
        )?;
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
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
        "ragged async LFM2 query must have shape [N,32,64]"
    );
    ensure!(block_tables.rank() == 2, "block tables must be rank 2");
    ensure!(
        block_tables.dims()[1] == block_table_stride,
        "block table stride/shape mismatch"
    );
    let num_tokens = query.dims()[0];
    ensure!(
        request_slots.numel() == num_tokens,
        "ragged async request slot count mismatch"
    );
    ensure!(
        position_ids.numel() == num_tokens,
        "ragged async position count mismatch"
    );

    let mut output = runtime.alloc_bf16(Shape::new([num_tokens, 32, 64]))?;
    unsafe {
        runtime
            .kernels()
            .attention_async()
            .launch_ragged_lfm2_bf16(
                runtime.stream(),
                arena.page_size().value(),
                query.storage(),
                arena.key().storage(),
                arena.value().storage(),
                block_tables.storage(),
                request_slots.storage(),
                position_ids.storage(),
                output.storage_mut(),
                num_tokens,
                arena.num_pages(),
                block_table_stride,
            )?;
    }
    Ok(output)
}
