use anyhow::Result;
use half::bf16;

use crate::{
    cache::{KvPageSize, PagedKvArena, PagedKvCache},
    cuda::{
        CudaRuntime,
        benchmark::{BenchConfig, benchmark_gpu_paired},
        testing::{assert_close_bf16, readback},
    },
    tensor::Shape,
};

fn bf16_values(elements: usize, mul: usize, modulus: usize, center: f32, scale: f32) -> Vec<bf16> {
    (0..elements)
        .map(|index| bf16::from_f32(((index * mul % modulus) as f32 - center) / scale))
        .collect()
}

fn seed_cache(runtime: &CudaRuntime, cache: &mut PagedKvCache, context: usize) -> Result<()> {
    let key = runtime.upload(
        &bf16_values(context * 8 * 64, 13, 89, 44.0, 64.0),
        Shape::new([context, 8, 64]),
    )?;
    let value = runtime.upload(
        &bf16_values(context * 8 * 64, 7, 79, 39.0, 32.0),
        Shape::new([context, 8, 64]),
    )?;
    let slots_host = (0..context)
        .map(i64::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let slots = runtime.upload(&slots_host, Shape::new([context]))?;
    cache.write_lfm2(runtime, &key, &value, &slots)
}

fn check_single_oneexp_matches_w8(page_size: KvPageSize) -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let page = page_size.value();
    let query = runtime.upload(
        &bf16_values(32 * 64, 17, 101, 50.0, 64.0),
        Shape::new([1, 32, 64]),
    )?;

    for context in [1usize, page, page + 1, page * 2 + 1, page * 8 + 3] {
        let mut cache = PagedKvCache::new(&runtime, context, page_size)?;
        seed_cache(&runtime, &mut cache, context)?;
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
            runtime.kernels().attention_async_oneexp().launch_lfm2_bf16(
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

#[test]
fn async_w8_one_exp_ps16_matches_two_exp() -> Result<()> {
    check_single_oneexp_matches_w8(KvPageSize::P16)
}

#[test]
fn async_w8_one_exp_ps32_matches_two_exp() -> Result<()> {
    check_single_oneexp_matches_w8(KvPageSize::P32)
}

fn check_ragged_oneexp_matches_w8(page_size: KvPageSize) -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    const REQUESTS: usize = 2;
    let page = page_size.value();
    let context = page + 1;
    let pages_per_request = context.div_ceil(page);
    let total_pages = REQUESTS * pages_per_request;

    let mut block_tables_host = Vec::with_capacity(total_pages);
    for request in 0..REQUESTS {
        let base = request * pages_per_request;
        for logical in 0..pages_per_request {
            block_tables_host.push(u32::try_from(base + pages_per_request - 1 - logical)?);
        }
    }

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

    let tokens = REQUESTS * context;
    let key = runtime.upload(
        &bf16_values(tokens * 8 * 64, 13, 89, 44.0, 64.0),
        Shape::new([tokens, 8, 64]),
    )?;
    let value = runtime.upload(
        &bf16_values(tokens * 8 * 64, 7, 79, 39.0, 32.0),
        Shape::new([tokens, 8, 64]),
    )?;
    let slots = runtime.upload(&physical_slots_host, Shape::new([tokens]))?;
    let mut arena = PagedKvArena::new(&runtime, total_pages, page_size)?;
    arena.write_lfm2(&runtime, &key, &value, &slots)?;

    let query = runtime.upload(
        &bf16_values(REQUESTS * 32 * 64, 17, 101, 50.0, 64.0),
        Shape::new([REQUESTS, 32, 64]),
    )?;
    let block_tables = runtime.upload(
        &block_tables_host,
        Shape::new([REQUESTS, pages_per_request]),
    )?;
    let request_slots = runtime.upload(&[0u32, 1u32], Shape::new([REQUESTS]))?;
    let positions = runtime.upload(
        &[u32::try_from(context - 1)?, u32::try_from(context - 1)?],
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
        runtime.kernels().attention_async_oneexp().launch_ragged_lfm2_bf16(
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
fn async_w8_one_exp_ragged_ps16_matches_two_exp() -> Result<()> {
    check_ragged_oneexp_matches_w8(KvPageSize::P16)
}

#[test]
fn async_w8_one_exp_ragged_ps32_matches_two_exp() -> Result<()> {
    check_ragged_oneexp_matches_w8(KvPageSize::P32)
}

#[test]
#[ignore = "GPU benchmark"]
fn bench_mok_async_w8_one_exp_paired_ab() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let query = runtime.upload(
        &bf16_values(32 * 64, 17, 101, 50.0, 64.0),
        Shape::new([1, 32, 64]),
    )?;
    let config = BenchConfig {
        warmup: 20,
        batches: 60,
        iterations_per_batch: 20,
    };

    for page_size in [KvPageSize::P16, KvPageSize::P32] {
        for context in [16usize, 32, 128, 512, 2048, 8192] {
            let mut cache = PagedKvCache::new(&runtime, context, page_size)?;
            seed_cache(&runtime, &mut cache, context)?;
            let position = runtime.upload(&[u32::try_from(context - 1)?], Shape::new([1]))?;
            let mut two_exp_output = runtime.alloc_bf16(Shape::new([1, 32, 64]))?;
            let mut one_exp_output = runtime.alloc_bf16(Shape::new([1, 32, 64]))?;

            let paired = benchmark_gpu_paired(
                runtime.context(),
                runtime.stream(),
                config,
                || {
                    unsafe {
                        runtime.kernels().attention_async().launch_lfm2_bf16(
                            runtime.stream(),
                            page_size.value(),
                            query.storage(),
                            cache.key().storage(),
                            cache.value().storage(),
                            cache.block_table().storage(),
                            position.storage(),
                            two_exp_output.storage_mut(),
                            1,
                            cache.num_pages(),
                        )?;
                    }
                    Ok(())
                },
                || {
                    unsafe {
                        runtime.kernels().attention_async_oneexp().launch_lfm2_bf16(
                            runtime.stream(),
                            page_size.value(),
                            query.storage(),
                            cache.key().storage(),
                            cache.value().storage(),
                            cache.block_table().storage(),
                            position.storage(),
                            one_exp_output.storage_mut(),
                            1,
                            cache.num_pages(),
                        )?;
                    }
                    Ok(())
                },
            )?;

            println!(
                "mok_async_w8_oneexp page_size={} context={} two_exp_mean={:.3}us two_exp_p50={:.3}us two_exp_p95={:.3}us one_exp_mean={:.3}us one_exp_p50={:.3}us one_exp_p95={:.3}us paired_speedup_mean={:.4}x paired_speedup_p50={:.4}x paired_speedup_p95={:.4}x speedup_min={:.4}x speedup_max={:.4}x",
                page_size.value(),
                context,
                paired.reference.mean_us,
                paired.reference.p50_us,
                paired.reference.p95_us,
                paired.candidate.mean_us,
                paired.candidate.p50_us,
                paired.candidate.p95_us,
                paired.speedup_mean,
                paired.speedup_p50,
                paired.speedup_p95,
                paired.speedup_min,
                paired.speedup_max,
            );
        }
    }
    Ok(())
}
