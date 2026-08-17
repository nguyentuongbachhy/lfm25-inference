#[cfg(test)]
use anyhow::Context as _;
use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cache::{PagedKvArena, PagedKvCache},
    cuda::CudaRuntime,
    tensor::{Shape, Tensor},
};

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
        runtime.kernels().attention().launch_prefill_lfm2_bf16(
            runtime.stream(),
            query.storage(),
            key.storage(),
            value.storage(),
            output.storage_mut(),
            num_tokens,
        )?;
    }
    Ok(output)
}

#[cfg(test)]
pub fn paged_ragged_attention_lfm2_bf16(
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
            block_table_stride,
        )?;
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn hybrid_ragged_attention_lfm2_bf16(
    runtime: &CudaRuntime,
    query: &Tensor<bf16>,
    current_key: &Tensor<bf16>,
    current_value: &Tensor<bf16>,
    arena: &PagedKvArena,
    block_tables: &Tensor<u32>,
    block_table_stride: usize,
    request_slots: &Tensor<u32>,
    position_ids: &Tensor<u32>,
    segment_offsets: &Tensor<u32>,
) -> Result<Tensor<bf16>> {
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
                arena.page_size().value(),
                query.storage(),
                current_key.storage(),
                current_value.storage(),
                arena.key().storage(),
                arena.value().storage(),
                block_tables.storage(),
                request_slots.storage(),
                position_ids.storage(),
                segment_offsets.storage(),
                output.storage_mut(),
                num_tokens,
                arena.num_pages(),
                block_table_stride,
                num_segments,
            )?;
    }
    Ok(output)
}

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

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::{
        cache::KvPageSize,
        cuda::{
            benchmark::{BenchConfig, benchmark_gpu},
            testing::{assert_close_bf16, readback},
        },
        ops::attention_async::paged_attention_lfm2_bf16 as paged_attention_async,
    };

    use super::*;

    fn make_inputs(tokens: usize) -> (Vec<bf16>, Vec<bf16>, Vec<bf16>) {
        let query = (0..tokens * 32 * 64)
            .map(|index| bf16::from_f32(((index * 17 % 101) as f32 - 50.0) / 64.0))
            .collect();
        let key = (0..tokens * 8 * 64)
            .map(|index| bf16::from_f32(((index * 13 % 89) as f32 - 44.0) / 64.0))
            .collect();
        let value = (0..tokens * 8 * 64)
            .map(|index| bf16::from_f32(((index * 7 % 79) as f32 - 39.0) / 32.0))
            .collect();
        (query, key, value)
    }

    fn check_causal(page_size: KvPageSize) -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let query = runtime.zeros::<bf16>(Shape::new([2, 32, 64]))?;
        let key = runtime.zeros::<bf16>(Shape::new([2, 8, 64]))?;
        let mut values = vec![bf16::from_f32(0.0); 2 * 8 * 64];
        for token in 0..2 {
            for head in 0..8 {
                for dim in 0..64 {
                    values[(token * 8 + head) * 64 + dim] =
                        bf16::from_f32((token * 10 + head) as f32 + dim as f32 / 64.0);
                }
            }
        }
        let value = runtime.upload(&values, Shape::new([2, 8, 64]))?;
        let slots = runtime.upload(&[0i64, 1], Shape::new([2]))?;
        let positions = runtime.upload(&[0u32, 1], Shape::new([2]))?;
        let mut cache = PagedKvCache::new(&runtime, page_size.value(), page_size)?;
        cache.write_lfm2(&runtime, &key, &value, &slots)?;

        let output = paged_attention_lfm2_bf16_sync(&runtime, &query, &cache, &positions)?;
        let actual = readback(&runtime, &output)?;
        let mut expected = Vec::with_capacity(2 * 32 * 64);
        for token in 0..2 {
            for query_head in 0..32 {
                let kv_head = query_head / 4;
                for dim in 0..64 {
                    let first = values[kv_head * 64 + dim].to_f32();
                    let value = if token == 0 {
                        first
                    } else {
                        let second = values[(8 + kv_head) * 64 + dim].to_f32();
                        (first + second) / 2.0
                    };
                    expected.push(bf16::from_f32(value));
                }
            }
        }
        assert_close_bf16(&actual, &expected, 0.02, 0.01);
        Ok(())
    }

    #[test]
    fn contiguous_prefill_is_causal_for_zero_queries() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let query = runtime.zeros::<bf16>(Shape::new([2, 32, 64]))?;
        let key = runtime.zeros::<bf16>(Shape::new([2, 8, 64]))?;
        let mut values = vec![bf16::from_f32(0.0); 2 * 8 * 64];
        for token in 0..2 {
            for head in 0..8 {
                for dim in 0..64 {
                    values[(token * 8 + head) * 64 + dim] =
                        bf16::from_f32((token * 10 + head) as f32 + dim as f32 / 64.0);
                }
            }
        }
        let value = runtime.upload(&values, Shape::new([2, 8, 64]))?;
        let output = prefill_attention_lfm2_bf16(&runtime, &query, &key, &value)?;
        let actual = readback(&runtime, &output)?;
        let mut expected = Vec::with_capacity(2 * 32 * 64);
        for token in 0..2 {
            for query_head in 0..32 {
                let kv_head = query_head / 4;
                for dim in 0..64 {
                    let first = values[kv_head * 64 + dim].to_f32();
                    let result = if token == 0 {
                        first
                    } else {
                        (first + values[(8 + kv_head) * 64 + dim].to_f32()) / 2.0
                    };
                    expected.push(bf16::from_f32(result));
                }
            }
        }
        assert_close_bf16(&actual, &expected, 0.02, 0.01);
        Ok(())
    }

    #[test]
    fn paged_attention_ps16_is_causal_for_zero_queries() -> Result<()> {
        check_causal(KvPageSize::P16)
    }

    #[test]
    fn paged_attention_ps32_is_causal_for_zero_queries() -> Result<()> {
        check_causal(KvPageSize::P32)
    }

    #[test]
    fn paged_attention_async_matches_sync_page_boundaries() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        for page_size in [KvPageSize::P16, KvPageSize::P32] {
            for tokens in [1usize, 15, 16, 17, 31, 32, 33, 127, 128, 511] {
                let (query_host, key_host, value_host) = make_inputs(tokens);
                let query = runtime.upload(
                    &query_host[(tokens - 1) * 32 * 64..],
                    Shape::new([1, 32, 64]),
                )?;
                let key = runtime.upload(&key_host, Shape::new([tokens, 8, 64]))?;
                let value = runtime.upload(&value_host, Shape::new([tokens, 8, 64]))?;
                let slots_host: Vec<i64> = (0..tokens)
                    .map(i64::try_from)
                    .collect::<std::result::Result<_, _>>()?;
                let slots = runtime.upload(&slots_host, Shape::new([tokens]))?;
                let position = runtime.upload(&[(tokens - 1) as u32], Shape::new([1]))?;
                let mut cache = PagedKvCache::new(&runtime, tokens, page_size)?;
                cache.write_lfm2(&runtime, &key, &value, &slots)?;

                let sync = paged_attention_lfm2_bf16_sync(&runtime, &query, &cache, &position)?;
                let async_output = paged_attention_async(&runtime, &query, &cache, &position)?;
                assert_close_bf16(
                    &readback(&runtime, &async_output)?,
                    &readback(&runtime, &sync)?,
                    0.03,
                    0.02,
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "GPU benchmark"]
    fn bench_paged_attention_sync_vs_async_lfm2_bf16() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let query = runtime.upload(
            &vec![bf16::from_f32(0.01); 32 * 64],
            Shape::new([1, 32, 64]),
        )?;
        let config = BenchConfig {
            warmup: 30,
            batches: 40,
            iterations_per_batch: 25,
        };

        for page_size in [KvPageSize::P16, KvPageSize::P32] {
            for sequence_length in [16usize, 32, 128, 512, 2048, 8192] {
                let cache = PagedKvCache::new(&runtime, sequence_length, page_size)?;
                let position = runtime.upload(&[(sequence_length - 1) as u32], Shape::new([1]))?;
                let mut sync_output = runtime.alloc_bf16(Shape::new([1, 32, 64]))?;
                let mut async_output = runtime.alloc_bf16(Shape::new([1, 32, 64]))?;

                let sync = benchmark_gpu(runtime.context(), runtime.stream(), config, || {
                    unsafe {
                        runtime.kernels().attention().launch_lfm2_bf16(
                            runtime.stream(),
                            page_size.value(),
                            query.storage(),
                            cache.key().storage(),
                            cache.value().storage(),
                            cache.block_table().storage(),
                            position.storage(),
                            sync_output.storage_mut(),
                            1,
                            cache.num_pages(),
                        )?;
                    }
                    Ok(())
                })?;

                let async_stats = benchmark_gpu(runtime.context(), runtime.stream(), config, || {
                    unsafe {
                        runtime.kernels().attention_async().launch_lfm2_bf16(
                            runtime.stream(),
                            page_size.value(),
                            query.storage(),
                            cache.key().storage(),
                            cache.value().storage(),
                            cache.block_table().storage(),
                            position.storage(),
                            async_output.storage_mut(),
                            1,
                            cache.num_pages(),
                        )?;
                    }
                    Ok(())
                })?;

                println!(
                    "page_size={} sequence_length={} sync_mean={:.3}us sync_p50={:.3}us sync_p95={:.3}us async_mean={:.3}us async_p50={:.3}us async_p95={:.3}us speedup={:.3}x",
                    page_size.value(),
                    sequence_length,
                    sync.mean_us,
                    sync.p50_us,
                    sync.p95_us,
                    async_stats.mean_us,
                    async_stats.p50_us,
                    async_stats.p95_us,
                    sync.mean_us / async_stats.mean_us,
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "GPU benchmark"]
    fn bench_prefill_attention_lfm2_bf16() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let config = BenchConfig {
            warmup: 10,
            batches: 20,
            iterations_per_batch: 10,
        };
        for num_tokens in [16usize, 128, 512, 1024] {
            let query = runtime.zeros::<bf16>(Shape::new([num_tokens, 32, 64]))?;
            let key = runtime.zeros::<bf16>(Shape::new([num_tokens, 8, 64]))?;
            let value = runtime.zeros::<bf16>(Shape::new([num_tokens, 8, 64]))?;
            let mut output = runtime.zeros::<bf16>(Shape::new([num_tokens, 32, 64]))?;
            let stats = benchmark_gpu(runtime.context(), runtime.stream(), config, || {
                unsafe {
                    runtime.kernels().attention().launch_prefill_lfm2_bf16(
                        runtime.stream(),
                        query.storage(),
                        key.storage(),
                        value.storage(),
                        output.storage_mut(),
                        num_tokens,
                    )?;
                }
                Ok(())
            })?;
            println!(
                "num_tokens={} mean={:.3}us p50={:.3}us p95={:.3}us min={:.3}us",
                num_tokens, stats.mean_us, stats.p50_us, stats.p95_us, stats.min_us,
            );
        }
        Ok(())
    }
}
