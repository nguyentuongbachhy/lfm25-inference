use anyhow::{Context as _, Result};
use half::bf16;

use crate::{
    cache::{KvPageSize, PagedKvArena},
    cuda::{
        CudaRuntime, FastRaggedAttentionLaunch, SplitKRaggedAttentionLaunch,
        benchmark::{BenchConfig, benchmark_gpu_paired},
    },
    tensor::Shape,
};

use super::attention_async_fast::{splitk_decode_splits, splitk_workspace_elements};

fn bench_splitk_case(
    runtime: &CudaRuntime,
    page_size: KvPageSize,
    config: BenchConfig,
    batch: usize,
    context: usize,
) -> Result<()> {
    let query = runtime.upload(
        &vec![bf16::from_f32(0.01); batch * 32 * 64],
        Shape::new([batch, 32, 64]),
    )?;
    let request_slots_host = (0..batch)
        .map(u32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let request_slots = runtime.upload(&request_slots_host, Shape::new([batch]))?;

    let pages_per_request = context.div_ceil(page_size.value());
    let total_pages = pages_per_request
        .checked_mul(batch)
        .context("split-K benchmark page count overflow")?;
    let arena = PagedKvArena::new(runtime, total_pages, page_size)?;

    let mut block_table_host = vec![u32::MAX; batch * pages_per_request];
    for request in 0..batch {
        for logical_page in 0..pages_per_request {
            block_table_host[request * pages_per_request + logical_page] = u32::try_from(
                request
                    .checked_mul(pages_per_request)
                    .and_then(|base| base.checked_add(logical_page))
                    .context("split-K benchmark physical page overflow")?,
            )?;
        }
    }
    let block_tables = runtime.upload(
        &block_table_host,
        Shape::new([batch, pages_per_request]),
    )?;
    let positions = runtime.upload(
        &vec![u32::try_from(context - 1)?; batch],
        Shape::new([batch]),
    )?;
    let production_splits = splitk_decode_splits(batch, context, page_size.value());

    for splits in [2usize, 4, 8] {
        let mut baseline_output = runtime.alloc_bf16(Shape::new([batch, 32, 64]))?;
        let mut split_output = runtime.alloc_bf16(Shape::new([batch, 32, 64]))?;
        let mut partials = runtime.alloc_uninit::<f32>(Shape::new([
            splitk_workspace_elements(batch)?,
        ]))?;

        let paired = benchmark_gpu_paired(
            runtime.context(),
            runtime.stream(),
            config,
            || {
                unsafe {
                    runtime
                        .kernels()
                        .attention_async_fast()
                        .launch_ragged_lfm2_bf16(
                            runtime.stream(),
                            FastRaggedAttentionLaunch {
                                page_size: page_size.value(),
                                query: query.storage(),
                                key_cache: arena.key().storage(),
                                value_cache: arena.value().storage(),
                                block_tables: block_tables.storage(),
                                request_slots: request_slots.storage(),
                                position_ids: positions.storage(),
                                output: baseline_output.storage_mut(),
                                num_tokens: batch,
                                num_pages: arena.num_pages(),
                                block_table_stride: pages_per_request,
                            },
                        )?;
                }
                Ok(())
            },
            || {
                unsafe {
                    runtime
                        .kernels()
                        .attention_async_fast()
                        .launch_splitk_ragged_lfm2_bf16(
                            runtime.stream(),
                            SplitKRaggedAttentionLaunch {
                                page_size: page_size.value(),
                                query: query.storage(),
                                key_cache: arena.key().storage(),
                                value_cache: arena.value().storage(),
                                block_tables: block_tables.storage(),
                                request_slots: request_slots.storage(),
                                position_ids: positions.storage(),
                                partials: partials.storage_mut(),
                                output: split_output.storage_mut(),
                                num_tokens: batch,
                                num_pages: arena.num_pages(),
                                block_table_stride: pages_per_request,
                                num_splits: splits,
                            },
                        )?;
                }
                Ok(())
            },
        )?;

        println!(
            "splitk_sweep page_size={} batch={} context={} splits={} production_splits={} baseline_mean={:.3}us baseline_p50={:.3}us baseline_p95={:.3}us split_mean={:.3}us split_p50={:.3}us split_p95={:.3}us speedup_mean={:.4}x speedup_p50={:.4}x speedup_p95={:.4}x speedup_min={:.4}x speedup_max={:.4}x",
            page_size.value(),
            batch,
            context,
            splits,
            production_splits,
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
    Ok(())
}

#[test]
#[ignore = "GPU benchmark"]
fn bench_splitk_decode_sweep_paired() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let page_size = KvPageSize::P16;
    let config = BenchConfig {
        warmup: 10,
        batches: 30,
        iterations_per_batch: 20,
    };

    for batch in [1usize, 2, 4] {
        for context in [512usize, 1024, 2048, 4096, 8192] {
            bench_splitk_case(&runtime, page_size, config, batch, context)?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "GPU benchmark"]
fn bench_splitk_decode_high_batch_sweep_paired() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let page_size = KvPageSize::P16;
    let config = BenchConfig {
        warmup: 8,
        batches: 20,
        iterations_per_batch: 10,
    };

    for (batch, contexts) in [
        (8usize, &[512usize, 1024, 2048, 4096, 8192][..]),
        (16usize, &[512usize, 1024, 2048, 4096][..]),
        (32usize, &[512usize, 1024, 2048][..]),
        (64usize, &[512usize, 1024, 2048][..]),
    ] {
        for &context in contexts {
            bench_splitk_case(&runtime, page_size, config, batch, context)?;
        }
    }
    Ok(())
}
