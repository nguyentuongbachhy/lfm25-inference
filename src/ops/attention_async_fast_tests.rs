use anyhow::Result;
use half::bf16;

use crate::{
    cache::{KvPageSize, PagedKvArena, PagedKvCache},
    cuda::{
        CudaRuntime,
        testing::{assert_close_bf16, readback},
    },
    tensor::Shape,
};

use super::{
    attention::{paged_attention_lfm2_bf16_sync, paged_ragged_attention_lfm2_bf16},
    attention_async_fast::{
        FastRaggedAttentionInput, paged_attention_fast_lfm2_bf16,
        paged_ragged_attention_fast_lfm2_bf16,
    },
};

fn bf16_values(elements: usize, mul: usize, modulus: usize, center: f32, scale: f32) -> Vec<bf16> {
    (0..elements)
        .map(|index| bf16::from_f32(((index * mul % modulus) as f32 - center) / scale))
        .collect()
}

fn check_single_fast_exp_matches_reference(page_size: KvPageSize) -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let page = page_size.value();

    for context in [1usize, page, page + 1, page * 2 + 1, page * 4 + 3, 128] {
        let mut cache = PagedKvCache::new(&runtime, context, page_size)?;
        let key = runtime.upload(
            &bf16_values(context * 8 * 64, 13 + context, 89, 44.0, 64.0),
            Shape::new([context, 8, 64]),
        )?;
        let value = runtime.upload(
            &bf16_values(context * 8 * 64, 7 + context, 79, 39.0, 32.0),
            Shape::new([context, 8, 64]),
        )?;
        let slots_host = (0..context)
            .map(i64::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let slots = runtime.upload(&slots_host, Shape::new([context]))?;
        cache.write_lfm2(&runtime, &key, &value, &slots)?;

        let query = runtime.upload(
            &bf16_values(32 * 64, 17 + context, 101, 50.0, 64.0),
            Shape::new([1, 32, 64]),
        )?;
        let position = runtime.upload(&[u32::try_from(context - 1)?], Shape::new([1]))?;
        let reference = paged_attention_lfm2_bf16_sync(&runtime, &query, &cache, &position)?;
        let candidate = paged_attention_fast_lfm2_bf16(&runtime, &query, &cache, &position)?;

        assert_close_bf16(
            &readback(&runtime, &candidate)?,
            &readback(&runtime, &reference)?,
            0.035,
            0.025,
        );
    }
    Ok(())
}

fn check_ragged_fast_exp_matches_reference(page_size: KvPageSize) -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    const REQUESTS: usize = 2;
    let page = page_size.value();
    let context = page * 2 + 1;
    let pages_per_request = context.div_ceil(page);
    let total_pages = REQUESTS * pages_per_request;

    let mut block_tables_host = Vec::with_capacity(total_pages);
    for request in 0..REQUESTS {
        let base = request * pages_per_request;
        for logical in 0..pages_per_request {
            block_tables_host.push(u32::try_from(base + pages_per_request - 1 - logical)?);
        }
    }
    let block_tables = runtime.upload(
        &block_tables_host,
        Shape::new([REQUESTS, pages_per_request]),
    )?;

    let mut physical_slots_host = Vec::with_capacity(REQUESTS * context);
    for request in 0..REQUESTS {
        for position in 0..context {
            let logical_page = position / page;
            let offset = position % page;
            let table_index = request * pages_per_request + logical_page;
            let physical_page = usize::try_from(block_tables_host[table_index])?;
            physical_slots_host.push(i64::try_from(physical_page * page + offset)?);
        }
    }

    let key = runtime.upload(
        &bf16_values(REQUESTS * context * 8 * 64, 13, 89, 44.0, 64.0),
        Shape::new([REQUESTS * context, 8, 64]),
    )?;
    let value = runtime.upload(
        &bf16_values(REQUESTS * context * 8 * 64, 7, 79, 39.0, 32.0),
        Shape::new([REQUESTS * context, 8, 64]),
    )?;
    let physical_slots = runtime.upload(&physical_slots_host, Shape::new([REQUESTS * context]))?;
    let mut arena = PagedKvArena::new(&runtime, total_pages, page_size)?;
    arena.write_lfm2(&runtime, &key, &value, &physical_slots)?;

    let query = runtime.upload(
        &bf16_values(REQUESTS * 32 * 64, 17, 101, 50.0, 64.0),
        Shape::new([REQUESTS, 32, 64]),
    )?;
    let request_slots = runtime.upload(&[0u32, 1u32], Shape::new([REQUESTS]))?;
    let positions = runtime.upload(
        &[u32::try_from(context - 1)?, u32::try_from(context - 1)?],
        Shape::new([REQUESTS]),
    )?;
    let reference = paged_ragged_attention_lfm2_bf16(
        &runtime,
        &query,
        &arena,
        &block_tables,
        pages_per_request,
        &request_slots,
        &positions,
    )?;
    let candidate = paged_ragged_attention_fast_lfm2_bf16(
        &runtime,
        FastRaggedAttentionInput {
            query: &query,
            arena: &arena,
            block_tables: &block_tables,
            block_table_stride: pages_per_request,
            request_slots: &request_slots,
            position_ids: &positions,
        },
    )?;

    assert_close_bf16(
        &readback(&runtime, &candidate)?,
        &readback(&runtime, &reference)?,
        0.035,
        0.025,
    );
    Ok(())
}

#[test]
fn async_w8_fast_exp_ps16_matches_reference() -> Result<()> {
    check_single_fast_exp_matches_reference(KvPageSize::P16)
}

#[test]
fn async_w8_fast_exp_ps32_matches_reference() -> Result<()> {
    check_single_fast_exp_matches_reference(KvPageSize::P32)
}

#[test]
fn async_w8_fast_exp_ragged_ps16_matches_reference() -> Result<()> {
    check_ragged_fast_exp_matches_reference(KvPageSize::P16)
}

#[test]
fn async_w8_fast_exp_ragged_ps32_matches_reference() -> Result<()> {
    check_ragged_fast_exp_matches_reference(KvPageSize::P32)
}
