use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cache::{PagedKvArena, PagedKvCache},
    cuda::CudaRuntime,
    tensor::{Shape, Tensor},
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_paged_attention_decode_lfm2_bf16(
    runtime: &CudaRuntime,
    query_raw: &Tensor<bf16>,
    key_raw: &Tensor<bf16>,
    value_raw: &Tensor<bf16>,
    query_norm: &Tensor<bf16>,
    key_norm: &Tensor<bf16>,
    inv_freq: &Tensor<f32>,
    position_ids: &Tensor<u32>,
    slot_mapping: &Tensor<i64>,
    cache: &mut PagedKvCache,
    eps: f32,
) -> Result<Tensor<bf16>> {
    validate_inputs(
        query_raw,
        key_raw,
        value_raw,
        query_norm,
        key_norm,
        inv_freq,
        position_ids,
        slot_mapping,
    )?;
    let num_tokens = query_raw.dims()[0];
    let page_size = cache.page_size().value();
    let num_pages = cache.num_pages();
    let mut output = runtime.alloc_bf16(Shape::new([num_tokens, 32, 64]))?;
    let (block_table, key_cache, value_cache) = cache.attention_parts_mut();

    unsafe {
        runtime.kernels().attention_fused().launch_decode(
            runtime.stream(),
            page_size,
            query_raw.storage(),
            key_raw.storage(),
            value_raw.storage(),
            query_norm.storage(),
            key_norm.storage(),
            inv_freq.storage(),
            key_cache.storage_mut(),
            value_cache.storage_mut(),
            block_table.storage(),
            position_ids.storage(),
            slot_mapping.storage(),
            output.storage_mut(),
            num_tokens,
            num_pages,
            eps,
        )?;
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_ragged_paged_attention_decode_lfm2_bf16(
    runtime: &CudaRuntime,
    query_raw: &Tensor<bf16>,
    key_raw: &Tensor<bf16>,
    value_raw: &Tensor<bf16>,
    query_norm: &Tensor<bf16>,
    key_norm: &Tensor<bf16>,
    inv_freq: &Tensor<f32>,
    block_tables: &Tensor<u32>,
    block_table_stride: usize,
    request_slots: &Tensor<u32>,
    position_ids: &Tensor<u32>,
    slot_mapping: &Tensor<i64>,
    arena: &mut PagedKvArena,
    eps: f32,
) -> Result<Tensor<bf16>> {
    validate_inputs(
        query_raw,
        key_raw,
        value_raw,
        query_norm,
        key_norm,
        inv_freq,
        position_ids,
        slot_mapping,
    )?;
    ensure!(block_tables.rank() == 2, "fused block tables must be rank 2");
    ensure!(
        block_tables.dims()[1] == block_table_stride,
        "fused block table stride/shape mismatch"
    );
    let num_tokens = query_raw.dims()[0];
    ensure!(
        request_slots.numel() == num_tokens,
        "fused request slot count mismatch"
    );

    let page_size = arena.page_size().value();
    let num_pages = arena.num_pages();
    let mut output = runtime.alloc_bf16(Shape::new([num_tokens, 32, 64]))?;
    let (key_cache, value_cache) = arena.kv_mut();
    unsafe {
        runtime
            .kernels()
            .attention_fused()
            .launch_ragged_decode(
                runtime.stream(),
                page_size,
                query_raw.storage(),
                key_raw.storage(),
                value_raw.storage(),
                query_norm.storage(),
                key_norm.storage(),
                inv_freq.storage(),
                key_cache.storage_mut(),
                value_cache.storage_mut(),
                block_tables.storage(),
                request_slots.storage(),
                position_ids.storage(),
                slot_mapping.storage(),
                output.storage_mut(),
                num_tokens,
                num_pages,
                block_table_stride,
                eps,
            )?;
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn validate_inputs(
    query_raw: &Tensor<bf16>,
    key_raw: &Tensor<bf16>,
    value_raw: &Tensor<bf16>,
    query_norm: &Tensor<bf16>,
    key_norm: &Tensor<bf16>,
    inv_freq: &Tensor<f32>,
    position_ids: &Tensor<u32>,
    slot_mapping: &Tensor<i64>,
) -> Result<()> {
    ensure!(
        query_raw.rank() == 3 && query_raw.dims()[1..] == [32, 64],
        "fused query must have shape [N,32,64]"
    );
    let num_tokens = query_raw.dims()[0];
    ensure!(
        key_raw.dims() == [num_tokens, 8, 64],
        "fused key must have shape [N,8,64]"
    );
    ensure!(value_raw.shape() == key_raw.shape(), "fused K/V shape mismatch");
    ensure!(query_norm.numel() == 64, "fused query norm must have 64 elements");
    ensure!(key_norm.numel() == 64, "fused key norm must have 64 elements");
    ensure!(inv_freq.numel() == 32, "fused RoPE inv_freq must have 32 elements");
    ensure!(position_ids.numel() == num_tokens, "fused position count mismatch");
    ensure!(slot_mapping.numel() == num_tokens, "fused slot count mismatch");
    Ok(())
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::{
        cache::KvPageSize,
        cuda::testing::{assert_close_bf16, readback},
        ops::{
            paged_attention_lfm2_bf16, paged_ragged_attention_lfm2_bf16,
            qk_norm_rope_kv_write_arena_decode_bf16,
            qk_norm_rope_kv_write_decode_bf16,
        },
    };

    use super::*;

    const EPS: f32 = 1.0e-5;

    fn bf16_values(elements: usize, mul: usize, modulus: usize, center: f32, scale: f32) -> Vec<bf16> {
        (0..elements)
            .map(|index| bf16::from_f32(((index * mul % modulus) as f32 - center) / scale))
            .collect()
    }

    fn norm_values(elements: usize, mul: usize) -> Vec<bf16> {
        (0..elements)
            .map(|index| bf16::from_f32(0.75 + ((index * mul % 29) as f32) / 64.0))
            .collect()
    }

    fn inv_freq_values() -> Vec<f32> {
        (0..32)
            .map(|index| 10_000.0f32.powf(-2.0 * index as f32 / 64.0))
            .collect()
    }

    fn seed_single_history(
        runtime: &CudaRuntime,
        cache: &mut PagedKvCache,
        history: usize,
    ) -> Result<()> {
        if history == 0 {
            return Ok(());
        }
        let key = runtime.upload(
            &bf16_values(history * 8 * 64, 11, 83, 41.0, 64.0),
            Shape::new([history, 8, 64]),
        )?;
        let value = runtime.upload(
            &bf16_values(history * 8 * 64, 7, 79, 39.0, 32.0),
            Shape::new([history, 8, 64]),
        )?;
        let slots_host = (0..history)
            .map(i64::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let slots = runtime.upload(&slots_host, Shape::new([history]))?;
        cache.write_lfm2(runtime, &key, &value, &slots)
    }

    fn check_single_matches_two_kernel(page_size: KvPageSize) -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let page = page_size.value();
        let query_norm = runtime.upload(&norm_values(64, 3), Shape::new([64]))?;
        let key_norm = runtime.upload(&norm_values(64, 5), Shape::new([64]))?;
        let inv_freq = runtime.upload(&inv_freq_values(), Shape::new([32]))?;

        for context in [1usize, page, page + 1, page * 2 + 1, page * 4 + 3] {
            let query_host = bf16_values(32 * 64, 17 + context, 101, 50.0, 64.0);
            let key_host = bf16_values(8 * 64, 13 + context, 89, 44.0, 64.0);
            let value_host = bf16_values(8 * 64, 7 + context, 79, 39.0, 32.0);
            let query_raw = runtime.upload(&query_host, Shape::new([1, 32, 64]))?;
            let key_raw = runtime.upload(&key_host, Shape::new([1, 8, 64]))?;
            let value_raw = runtime.upload(&value_host, Shape::new([1, 8, 64]))?;
            let position = runtime.upload(&[u32::try_from(context - 1)?], Shape::new([1]))?;
            let slot = runtime.upload(&[i64::try_from(context - 1)?], Shape::new([1]))?;

            let mut reference_cache = PagedKvCache::new(&runtime, context, page_size)?;
            let mut candidate_cache = PagedKvCache::new(&runtime, context, page_size)?;
            seed_single_history(&runtime, &mut reference_cache, context - 1)?;
            seed_single_history(&runtime, &mut candidate_cache, context - 1)?;

            let mut reference_query = runtime.upload(&query_host, Shape::new([1, 32, 64]))?;
            qk_norm_rope_kv_write_decode_bf16(
                &runtime,
                &mut reference_query,
                &key_raw,
                &value_raw,
                &query_norm,
                &key_norm,
                &inv_freq,
                &position,
                &slot,
                &mut reference_cache,
                EPS,
            )?;
            let reference = paged_attention_lfm2_bf16(
                &runtime,
                &reference_query,
                &reference_cache,
                &position,
            )?;
            let candidate = fused_paged_attention_decode_lfm2_bf16(
                &runtime,
                &query_raw,
                &key_raw,
                &value_raw,
                &query_norm,
                &key_norm,
                &inv_freq,
                &position,
                &slot,
                &mut candidate_cache,
                EPS,
            )?;

            assert_close_bf16(
                &readback(&runtime, &candidate)?,
                &readback(&runtime, &reference)?,
                0.035,
                0.025,
            );
            assert_close_bf16(
                &readback(&runtime, candidate_cache.key())?,
                &readback(&runtime, reference_cache.key())?,
                0.03,
                0.02,
            );
            assert_close_bf16(
                &readback(&runtime, candidate_cache.value())?,
                &readback(&runtime, reference_cache.value())?,
                0.0,
                0.0,
            );
        }
        Ok(())
    }

    fn check_ragged_matches_two_kernel(page_size: KvPageSize) -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        const REQUESTS: usize = 2;
        let page = page_size.value();
        let context = page + 1;
        let position_value = context - 1;
        let pages_per_request = context.div_ceil(page);
        let total_pages = REQUESTS * pages_per_request;

        let mut block_tables_host = Vec::with_capacity(total_pages);
        for request in 0..REQUESTS {
            let base = request * pages_per_request;
            for logical_page in 0..pages_per_request {
                block_tables_host.push(u32::try_from(base + pages_per_request - 1 - logical_page)?);
            }
        }
        let block_tables = runtime.upload(
            &block_tables_host,
            Shape::new([REQUESTS, pages_per_request]),
        )?;
        let request_slots = runtime.upload(&[0u32, 1u32], Shape::new([REQUESTS]))?;
        let positions = runtime.upload(
            &[u32::try_from(position_value)?, u32::try_from(position_value)?],
            Shape::new([REQUESTS]),
        )?;

        let mut history_slots = Vec::with_capacity(REQUESTS * position_value);
        for request in 0..REQUESTS {
            for position in 0..position_value {
                let logical_page = position / page;
                let offset = position % page;
                let physical_page = usize::try_from(
                    block_tables_host[request * pages_per_request + logical_page],
                )?;
                history_slots.push(i64::try_from(physical_page * page + offset)?);
            }
        }
        let history_tokens = history_slots.len();
        let history_key = runtime.upload(
            &bf16_values(history_tokens * 8 * 64, 11, 83, 41.0, 64.0),
            Shape::new([history_tokens, 8, 64]),
        )?;
        let history_value = runtime.upload(
            &bf16_values(history_tokens * 8 * 64, 7, 79, 39.0, 32.0),
            Shape::new([history_tokens, 8, 64]),
        )?;
        let history_slots = runtime.upload(&history_slots, Shape::new([history_tokens]))?;

        let mut current_slots_host = Vec::with_capacity(REQUESTS);
        for request in 0..REQUESTS {
            let logical_page = position_value / page;
            let offset = position_value % page;
            let physical_page = usize::try_from(
                block_tables_host[request * pages_per_request + logical_page],
            )?;
            current_slots_host.push(i64::try_from(physical_page * page + offset)?);
        }
        let current_slots = runtime.upload(&current_slots_host, Shape::new([REQUESTS]))?;
        let query_host = bf16_values(REQUESTS * 32 * 64, 17, 101, 50.0, 64.0);
        let key_host = bf16_values(REQUESTS * 8 * 64, 13, 89, 44.0, 64.0);
        let value_host = bf16_values(REQUESTS * 8 * 64, 5, 73, 36.0, 32.0);
        let query_raw = runtime.upload(&query_host, Shape::new([REQUESTS, 32, 64]))?;
        let key_raw = runtime.upload(&key_host, Shape::new([REQUESTS, 8, 64]))?;
        let value_raw = runtime.upload(&value_host, Shape::new([REQUESTS, 8, 64]))?;
        let query_norm = runtime.upload(&norm_values(64, 3), Shape::new([64]))?;
        let key_norm = runtime.upload(&norm_values(64, 5), Shape::new([64]))?;
        let inv_freq = runtime.upload(&inv_freq_values(), Shape::new([32]))?;

        let mut reference_arena = PagedKvArena::new(&runtime, total_pages, page_size)?;
        let mut candidate_arena = PagedKvArena::new(&runtime, total_pages, page_size)?;
        reference_arena.write_lfm2(&runtime, &history_key, &history_value, &history_slots)?;
        candidate_arena.write_lfm2(&runtime, &history_key, &history_value, &history_slots)?;

        let mut reference_query = runtime.upload(&query_host, Shape::new([REQUESTS, 32, 64]))?;
        qk_norm_rope_kv_write_arena_decode_bf16(
            &runtime,
            &mut reference_query,
            &key_raw,
            &value_raw,
            &query_norm,
            &key_norm,
            &inv_freq,
            &positions,
            &current_slots,
            &mut reference_arena,
            EPS,
        )?;
        let reference = paged_ragged_attention_lfm2_bf16(
            &runtime,
            &reference_query,
            &reference_arena,
            &block_tables,
            pages_per_request,
            &request_slots,
            &positions,
        )?;
        let candidate = fused_ragged_paged_attention_decode_lfm2_bf16(
            &runtime,
            &query_raw,
            &key_raw,
            &value_raw,
            &query_norm,
            &key_norm,
            &inv_freq,
            &block_tables,
            pages_per_request,
            &request_slots,
            &positions,
            &current_slots,
            &mut candidate_arena,
            EPS,
        )?;

        assert_close_bf16(
            &readback(&runtime, &candidate)?,
            &readback(&runtime, &reference)?,
            0.035,
            0.025,
        );
        assert_close_bf16(
            &readback(&runtime, candidate_arena.key())?,
            &readback(&runtime, reference_arena.key())?,
            0.03,
            0.02,
        );
        assert_close_bf16(
            &readback(&runtime, candidate_arena.value())?,
            &readback(&runtime, reference_arena.value())?,
            0.0,
            0.0,
        );
        Ok(())
    }

    #[test]
    fn fused_decode_attention_ps16_matches_two_kernel_mok() -> Result<()> {
        check_single_matches_two_kernel(KvPageSize::P16)
    }

    #[test]
    fn fused_decode_attention_ps32_matches_two_kernel_mok() -> Result<()> {
        check_single_matches_two_kernel(KvPageSize::P32)
    }

    #[test]
    fn fused_ragged_decode_attention_ps16_matches_two_kernel_mok() -> Result<()> {
        check_ragged_matches_two_kernel(KvPageSize::P16)
    }

    #[test]
    fn fused_ragged_decode_attention_ps32_matches_two_kernel_mok() -> Result<()> {
        check_ragged_matches_two_kernel(KvPageSize::P32)
    }
}
