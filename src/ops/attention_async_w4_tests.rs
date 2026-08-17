use anyhow::Result;
use half::bf16;

use crate::{
    cache::{KvPageSize, PagedKvArena, PagedKvCache},
    cuda::{testing::{assert_close_bf16, readback}, CudaRuntime},
    tensor::Shape,
};

fn query_values(tokens: usize) -> Vec<bf16> {
    (0..tokens * 32 * 64)
        .map(|index| bf16::from_f32(((index * 17 % 101) as f32 - 50.0) / 64.0))
        .collect()
}

fn kv_values(tokens: usize, mul: usize, modulus: usize, center: f32, scale: f32) -> Vec<bf16> {
    (0..tokens * 8 * 64)
        .map(|index| bf16::from_f32(((index * mul % modulus) as f32 - center) / scale))
        .collect()
}

fn check_single(page_size: KvPageSize) -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let page = page_size.value();

    for context in [1usize, page, page + 1, page * 2 + 1, 127, 128, 511] {
        let query = runtime.upload(&query_values(1), Shape::new([1, 32, 64]))?;
        let mut cache = PagedKvCache::new(&runtime, context, page_size)?;
        let key = runtime.upload(
            &kv_values(context, 13, 89, 44.0, 64.0),
            Shape::new([context, 8, 64]),
        )?;
        let value = runtime.upload(
            &kv_values(context, 7, 79, 39.0, 32.0),
            Shape::new([context, 8, 64]),
        )?;
        let slots_host = (0..context)
            .map(i64::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let slots = runtime.upload(&slots_host, Shape::new([context]))?;
        cache.write_lfm2(&runtime, &key, &value, &slots)?;
        let position = runtime.upload(&[u32::try_from(context - 1)?], Shape::new([1]))?;

        let mut reference = runtime.alloc_bf16(Shape::new([1, 32, 64]))?;
        let mut candidate = runtime.alloc_bf16(Shape::new([1, 32, 64]))?;
        unsafe {
            runtime.kernels().attention_async().launch_lfm2_bf16(
                runtime.stream(),
                page,
                query.storage(),
                cache.key().storage(),
                cache.value().storage(),
                cache.block_table().storage(),
                position.storage(),
                reference.storage_mut(),
                1,
                cache.num_pages(),
            )?;
            runtime.kernels().attention_async_w4().launch_lfm2_bf16(
                runtime.stream(),
                page,
                query.storage(),
                cache.key().storage(),
                cache.value().storage(),
                cache.block_table().storage(),
                position.storage(),
                candidate.storage_mut(),
                1,
                cache.num_pages(),
            )?;
        }

        assert_close_bf16(
            &readback(&runtime, &candidate)?,
            &readback(&runtime, &reference)?,
            0.03,
            0.02,
        );
    }
    Ok(())
}

fn check_ragged(page_size: KvPageSize) -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    const REQUESTS: usize = 2;
    const CONTEXT: usize = 33;
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
            let physical_page = usize::try_from(
                block_tables_host[request * pages_per_request + logical_page],
            )?;
            physical_slots_host.push(i64::try_from(physical_page * page + offset)?);
        }
    }

    let key = runtime.upload(
        &kv_values(REQUESTS * CONTEXT, 13, 89, 44.0, 64.0),
        Shape::new([REQUESTS * CONTEXT, 8, 64]),
    )?;
    let value = runtime.upload(
        &kv_values(REQUESTS * CONTEXT, 7, 79, 39.0, 32.0),
        Shape::new([REQUESTS * CONTEXT, 8, 64]),
    )?;
    let physical_slots = runtime.upload(
        &physical_slots_host,
        Shape::new([REQUESTS * CONTEXT]),
    )?;
    let mut arena = PagedKvArena::new(&runtime, total_pages, page_size)?;
    arena.write_lfm2(&runtime, &key, &value, &physical_slots)?;

    let query = runtime.upload(&query_values(REQUESTS), Shape::new([REQUESTS, 32, 64]))?;
    let block_tables = runtime.upload(
        &block_tables_host,
        Shape::new([REQUESTS, pages_per_request]),
    )?;
    let request_slots = runtime.upload(&[0u32, 1u32], Shape::new([REQUESTS]))?;
    let positions = runtime.upload(
        &[u32::try_from(CONTEXT - 1)?, u32::try_from(CONTEXT - 1)?],
        Shape::new([REQUESTS]),
    )?;
    let mut reference = runtime.alloc_bf16(Shape::new([REQUESTS, 32, 64]))?;
    let mut candidate = runtime.alloc_bf16(Shape::new([REQUESTS, 32, 64]))?;

    unsafe {
        runtime.kernels().attention_async().launch_ragged_lfm2_bf16(
            runtime.stream(),
            page,
            query.storage(),
            arena.key().storage(),
            arena.value().storage(),
            block_tables.storage(),
            request_slots.storage(),
            positions.storage(),
            reference.storage_mut(),
            REQUESTS,
            arena.num_pages(),
            pages_per_request,
        )?;
        runtime.kernels().attention_async_w4().launch_ragged_lfm2_bf16(
            runtime.stream(),
            page,
            query.storage(),
            arena.key().storage(),
            arena.value().storage(),
            block_tables.storage(),
            request_slots.storage(),
            positions.storage(),
            candidate.storage_mut(),
            REQUESTS,
            arena.num_pages(),
            pages_per_request,
        )?;
    }

    assert_close_bf16(
        &readback(&runtime, &candidate)?,
        &readback(&runtime, &reference)?,
        0.03,
        0.02,
    );
    Ok(())
}

#[test]
fn async_w4_one_exp_ps16_matches_w8_reference() -> Result<()> {
    check_single(KvPageSize::P16)
}

#[test]
fn async_w4_one_exp_ps32_matches_w8_reference() -> Result<()> {
    check_single(KvPageSize::P32)
}

#[test]
fn async_w4_one_exp_ragged_ps16_matches_w8_reference() -> Result<()> {
    check_ragged(KvPageSize::P16)
}

#[test]
fn async_w4_one_exp_ragged_ps32_matches_w8_reference() -> Result<()> {
    check_ragged(KvPageSize::P32)
}
