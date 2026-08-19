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

use super::attention::{
    HybridRaggedAttentionInput, hybrid_ragged_attention_lfm2_bf16, paged_attention_lfm2_bf16_sync,
    paged_ragged_attention_lfm2_bf16, prefill_attention_lfm2_bf16,
};

#[test]
fn ragged_attention_selects_each_request_block_table() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let query = runtime.zeros::<bf16>(Shape::new([2, 32, 64]))?;
    let key = runtime.zeros::<bf16>(Shape::new([2, 8, 64]))?;
    let mut values = vec![bf16::from_f32(0.0); 2 * 8 * 64];
    for token in 0..2 {
        for head in 0..8 {
            for dim in 0..64 {
                values[(token * 8 + head) * 64 + dim] =
                    bf16::from_f32((token * 100 + head * 2) as f32 + dim as f32 / 64.0);
            }
        }
    }
    let value = runtime.upload(&values, Shape::new([2, 8, 64]))?;
    let page_size = KvPageSize::P16;
    let mut arena = PagedKvArena::new(&runtime, 2, page_size)?;
    let physical_slots = runtime.upload(&[0i64, 16], Shape::new([2]))?;
    arena.write_lfm2(&runtime, &key, &value, &physical_slots)?;
    let block_tables = runtime.upload(&[0u32, 1], Shape::new([2, 1]))?;
    let request_slots = runtime.upload(&[0u32, 1], Shape::new([2]))?;
    let positions = runtime.upload(&[0u32, 0], Shape::new([2]))?;
    let output = paged_ragged_attention_lfm2_bf16(
        &runtime,
        &query,
        &arena,
        &block_tables,
        1,
        &request_slots,
        &positions,
    )?;
    let actual = readback(&runtime, &output)?;
    let mut expected = Vec::with_capacity(2 * 32 * 64);
    for token in 0..2 {
        for query_head in 0..32 {
            let kv_head = query_head / 4;
            expected.extend_from_slice(
                &values[(token * 8 + kv_head) * 64..(token * 8 + kv_head + 1) * 64],
            );
        }
    }
    assert_close_bf16(&actual, &expected, 0.02, 0.01);
    Ok(())
}

#[test]
fn hybrid_attention_uses_paged_prefix_and_contiguous_current_chunk() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let query = runtime.zeros::<bf16>(Shape::new([2, 32, 64]))?;
    let key = runtime.zeros::<bf16>(Shape::new([2, 8, 64]))?;
    let prefix_key = runtime.zeros::<bf16>(Shape::new([2, 8, 64]))?;
    let prefix_values = vec![bf16::from_f32(2.0); 2 * 8 * 64];
    let current_values = vec![bf16::from_f32(6.0); 2 * 8 * 64];
    let prefix_value = runtime.upload(&prefix_values, Shape::new([2, 8, 64]))?;
    let current_value = runtime.upload(&current_values, Shape::new([2, 8, 64]))?;
    let mut arena = PagedKvArena::new(&runtime, 1, KvPageSize::P16)?;
    let prefix_slots = runtime.upload(&[0i64, 1], Shape::new([2]))?;
    arena.write_lfm2(&runtime, &prefix_key, &prefix_value, &prefix_slots)?;
    let block_tables = runtime.upload(&[0u32], Shape::new([1, 1]))?;
    let request_slots = runtime.upload(&[0u32, 0], Shape::new([2]))?;
    let positions = runtime.upload(&[2u32, 3], Shape::new([2]))?;
    let segment_offsets = runtime.upload(&[0u32, 2], Shape::new([2]))?;
    let output = hybrid_ragged_attention_lfm2_bf16(
        &runtime,
        HybridRaggedAttentionInput {
            query: &query,
            current_key: &key,
            current_value: &current_value,
            arena: &arena,
            block_tables: &block_tables,
            block_table_stride: 1,
            request_slots: &request_slots,
            position_ids: &positions,
            segment_offsets: &segment_offsets,
        },
    )?;
    let actual = readback(&runtime, &output)?;
    let first_expected = (2.0 + 2.0 + 6.0) / 3.0;
    let second_expected = (2.0 + 2.0 + 6.0 + 6.0) / 4.0;
    for token in 0..2 {
        let expected = if token == 0 {
            first_expected
        } else {
            second_expected
        };
        for value in &actual[token * 32 * 64..(token + 1) * 32 * 64] {
            assert!((value.to_f32() - expected).abs() < 0.02);
        }
    }
    Ok(())
}

#[test]
fn paged_attention_matches_prefill_across_page_boundaries() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    const TOKENS: usize = 33;
    let query_host: Vec<bf16> = (0..TOKENS * 32 * 64)
        .map(|index| bf16::from_f32(((index * 17 % 101) as f32 - 50.0) / 64.0))
        .collect();
    let key_host: Vec<bf16> = (0..TOKENS * 8 * 64)
        .map(|index| bf16::from_f32(((index * 13 % 89) as f32 - 44.0) / 64.0))
        .collect();
    let value_host: Vec<bf16> = (0..TOKENS * 8 * 64)
        .map(|index| bf16::from_f32(((index * 7 % 79) as f32 - 39.0) / 32.0))
        .collect();
    let query = runtime.upload(&query_host, Shape::new([TOKENS, 32, 64]))?;
    let key = runtime.upload(&key_host, Shape::new([TOKENS, 8, 64]))?;
    let value = runtime.upload(&value_host, Shape::new([TOKENS, 8, 64]))?;
    let prefill = prefill_attention_lfm2_bf16(&runtime, &query, &key, &value)?;
    let prefill_host = readback(&runtime, &prefill)?;
    let query_last = runtime.upload(
        &query_host[(TOKENS - 1) * 32 * 64..],
        Shape::new([1, 32, 64]),
    )?;
    let slots_host: Vec<i64> = (0..TOKENS)
        .map(i64::try_from)
        .collect::<std::result::Result<_, _>>()?;
    let slots = runtime.upload(&slots_host, Shape::new([TOKENS]))?;
    let position = runtime.upload(&[(TOKENS - 1) as u32], Shape::new([1]))?;

    for page_size in [KvPageSize::P16, KvPageSize::P32] {
        let mut cache = PagedKvCache::new(&runtime, TOKENS, page_size)?;
        cache.write_lfm2(&runtime, &key, &value, &slots)?;
        let paged = paged_attention_lfm2_bf16_sync(&runtime, &query_last, &cache, &position)?;
        assert_close_bf16(
            &readback(&runtime, &paged)?,
            &prefill_host[(TOKENS - 1) * 32 * 64..],
            0.03,
            0.02,
        );
    }

    let page_size = KvPageSize::P16;
    let block_table = [2u32, 0, 1];
    let mapped_slots: Vec<i64> = (0..TOKENS)
        .map(|position| {
            let logical_page = position / page_size.value();
            let offset = position % page_size.value();
            let physical_page = usize::try_from(block_table[logical_page])?;
            let slot = physical_page
                .checked_mul(page_size.value())
                .and_then(|base| base.checked_add(offset))
                .ok_or_else(|| anyhow::anyhow!("mapped KV slot overflow"))?;
            Ok(i64::try_from(slot)?)
        })
        .collect::<Result<_>>()?;
    let mapped_slots = runtime.upload(&mapped_slots, Shape::new([TOKENS]))?;
    let mut cache = PagedKvCache::with_block_table(&runtime, page_size, 3, &block_table)?;
    cache.write_lfm2(&runtime, &key, &value, &mapped_slots)?;
    let paged = paged_attention_lfm2_bf16_sync(&runtime, &query_last, &cache, &position)?;
    assert_close_bf16(
        &readback(&runtime, &paged)?,
        &prefill_host[(TOKENS - 1) * 32 * 64..],
        0.03,
        0.02,
    );
    Ok(())
}
