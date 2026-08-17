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

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::{
        cache::KvPageSize,
        cuda::testing::{assert_close_bf16, readback},
        ops::attention::paged_ragged_attention_lfm2_bf16 as paged_ragged_sync,
    };

    use super::*;

    fn check_ragged_matches_sync(page_size: KvPageSize) -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        const CONTEXT: usize = 33;
        const REQUESTS: usize = 2;
        let page = page_size.value();
        let pages_per_request = CONTEXT.div_ceil(page);
        let total_pages = REQUESTS * pages_per_request;

        let mut block_tables_host = Vec::with_capacity(total_pages);
        for request in 0..REQUESTS {
            let base = request * pages_per_request;
            for logical in 0..pages_per_request {
                block_tables_host.push(u32::try_from(base + pages_per_request - 1 - logical)?);
            }
        }

        let mut physical_slots_host = Vec::with_capacity(REQUESTS * CONTEXT);
        for request in 0..REQUESTS {
            for position in 0..CONTEXT {
                let logical_page = position / page;
                let offset = position % page;
                let table_index = request * pages_per_request + logical_page;
                let physical_page = usize::try_from(block_tables_host[table_index])?;
                physical_slots_host.push(i64::try_from(physical_page * page + offset)?);
            }
        }

        let key_host: Vec<bf16> = (0..REQUESTS * CONTEXT * 8 * 64)
            .map(|index| bf16::from_f32(((index * 13 % 89) as f32 - 44.0) / 64.0))
            .collect();
        let value_host: Vec<bf16> = (0..REQUESTS * CONTEXT * 8 * 64)
            .map(|index| bf16::from_f32(((index * 7 % 79) as f32 - 39.0) / 32.0))
            .collect();
        let query_host: Vec<bf16> = (0..REQUESTS * 32 * 64)
            .map(|index| bf16::from_f32(((index * 17 % 101) as f32 - 50.0) / 64.0))
            .collect();

        let key = runtime.upload(
            &key_host,
            Shape::new([REQUESTS * CONTEXT, 8, 64]),
        )?;
        let value = runtime.upload(
            &value_host,
            Shape::new([REQUESTS * CONTEXT, 8, 64]),
        )?;
        let physical_slots = runtime.upload(
            &physical_slots_host,
            Shape::new([REQUESTS * CONTEXT]),
        )?;
        let mut arena = PagedKvArena::new(&runtime, total_pages, page_size)?;
        arena.write_lfm2(&runtime, &key, &value, &physical_slots)?;

        let query = runtime.upload(&query_host, Shape::new([REQUESTS, 32, 64]))?;
        let block_tables = runtime.upload(
            &block_tables_host,
            Shape::new([REQUESTS, pages_per_request]),
        )?;
        let request_slots = runtime.upload(&[0u32, 1], Shape::new([REQUESTS]))?;
        let positions = runtime.upload(
            &[u32::try_from(CONTEXT - 1)?, u32::try_from(CONTEXT - 1)?],
            Shape::new([REQUESTS]),
        )?;

        let reference = paged_ragged_sync(
            &runtime,
            &query,
            &arena,
            &block_tables,
            pages_per_request,
            &request_slots,
            &positions,
        )?;
        let candidate = paged_ragged_attention_lfm2_bf16(
            &runtime,
            &query,
            &arena,
            &block_tables,
            pages_per_request,
            &request_slots,
            &positions,
        )?;

        assert_close_bf16(
            &readback(&runtime, &candidate)?,
            &readback(&runtime, &reference)?,
            0.03,
            0.02,
        );
        Ok(())
    }

    #[test]
    fn async_ragged_paged_attention_ps16_matches_sync() -> Result<()> {
        check_ragged_matches_sync(KvPageSize::P16)
    }

    #[test]
    fn async_ragged_paged_attention_ps32_matches_sync() -> Result<()> {
        check_ragged_matches_sync(KvPageSize::P32)
    }
}
