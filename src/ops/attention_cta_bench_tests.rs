use anyhow::{Context as _, Result};
use half::bf16;

use crate::{
    cache::{KvPageSize, PagedKvArena},
    cuda::{
        CudaRuntime, FastRaggedAttentionLaunch, SplitKRaggedAttentionLaunch,
        benchmark::{BenchConfig, benchmark_gpu_paired},
        testing::{assert_eq_bf16, readback},
    },
    tensor::Shape,
};

use super::{
    attention_async_fast::splitk_workspace_elements,
    splitk_policy::splitk_decode_splits,
};

fn bf16_values(elements: usize, mul: usize, modulus: usize, center: f32, scale: f32) -> Vec<bf16> {
    (0..elements)
        .map(|index| bf16::from_f32(((index * mul % modulus) as f32 - center) / scale))
        .collect()
}

struct Fixture {
    query: crate::tensor::Tensor<bf16>,
    arena: PagedKvArena,
    block_tables: crate::tensor::Tensor<u32>,
    request_slots: crate::tensor::Tensor<u32>,
    positions: crate::tensor::Tensor<u32>,
    pages_per_request: usize,
}

fn fixture(
    runtime: &CudaRuntime,
    batch: usize,
    context: usize,
    nonzero_kv: bool,
) -> Result<Fixture> {
    let page_size = KvPageSize::P16;
    let page = page_size.value();
    let pages_per_request = context.div_ceil(page);
    let total_pages = pages_per_request
        .checked_mul(batch)
        .context("CTA geometry benchmark page count overflow")?;

    let mut block_table_host = vec![u32::MAX; batch * pages_per_request];
    for request in 0..batch {
        for logical_page in 0..pages_per_request {
            block_table_host[request * pages_per_request + logical_page] = u32::try_from(
                request
                    .checked_mul(pages_per_request)
                    .and_then(|base| base.checked_add(logical_page))
                    .context("CTA geometry physical page overflow")?,
            )?;
        }
    }
    let block_tables = runtime.upload(
        &block_table_host,
        Shape::new([batch, pages_per_request]),
    )?;
    let request_slots_host = (0..batch)
        .map(u32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let request_slots = runtime.upload(&request_slots_host, Shape::new([batch]))?;
    let positions = runtime.upload(
        &vec![u32::try_from(context - 1)?; batch],
        Shape::new([batch]),
    )?;
    let query = runtime.upload(
        &bf16_values(batch * 32 * 64, 17, 101, 50.0, 64.0),
        Shape::new([batch, 32, 64]),
    )?;

    let mut arena = PagedKvArena::new(runtime, total_pages, page_size)?;
    if nonzero_kv {
        let tokens = batch
            .checked_mul(context)
            .context("CTA geometry fixture token count overflow")?;
        let key = runtime.upload(
            &bf16_values(tokens * 8 * 64, 13, 89, 44.0, 64.0),
            Shape::new([tokens, 8, 64]),
        )?;
        let value = runtime.upload(
            &bf16_values(tokens * 8 * 64, 7, 79, 39.0, 32.0),
            Shape::new([tokens, 8, 64]),
        )?;
        let mut physical_slots_host = Vec::with_capacity(tokens);
        for request in 0..batch {
            for position in 0..context {
                let physical_page = request * pages_per_request + position / page;
                physical_slots_host.push(i64::try_from(physical_page * page + position % page)?);
            }
        }
        let physical_slots = runtime.upload(&physical_slots_host, Shape::new([tokens]))?;
        arena.write_lfm2(runtime, &key, &value, &physical_slots)?;
    }

    Ok(Fixture {
        query,
        arena,
        block_tables,
        request_slots,
        positions,
        pages_per_request,
    })
}

fn launch_256(
    runtime: &CudaRuntime,
    fixture: &Fixture,
    splits: usize,
    partials: &mut crate::tensor::Tensor<f32>,
    output: &mut crate::tensor::Tensor<bf16>,
) -> Result<()> {
    unsafe {
        if splits == 1 {
            runtime
                .kernels()
                .attention_async_fast()
                .launch_ragged_lfm2_bf16(
                    runtime.stream(),
                    FastRaggedAttentionLaunch {
                        page_size: 16,
                        query: fixture.query.storage(),
                        key_cache: fixture.arena.key().storage(),
                        value_cache: fixture.arena.value().storage(),
                        block_tables: fixture.block_tables.storage(),
                        request_slots: fixture.request_slots.storage(),
                        position_ids: fixture.positions.storage(),
                        output: output.storage_mut(),
                        num_tokens: fixture.query.dims()[0],
                        num_pages: fixture.arena.num_pages(),
                        block_table_stride: fixture.pages_per_request,
                    },
                )?;
        } else {
            runtime
                .kernels()
                .attention_async_fast()
                .launch_splitk_ragged_lfm2_bf16(
                    runtime.stream(),
                    SplitKRaggedAttentionLaunch {
                        page_size: 16,
                        query: fixture.query.storage(),
                        key_cache: fixture.arena.key().storage(),
                        value_cache: fixture.arena.value().storage(),
                        block_tables: fixture.block_tables.storage(),
                        request_slots: fixture.request_slots.storage(),
                        position_ids: fixture.positions.storage(),
                        partials: partials.storage_mut(),
                        output: output.storage_mut(),
                        num_tokens: fixture.query.dims()[0],
                        num_pages: fixture.arena.num_pages(),
                        block_table_stride: fixture.pages_per_request,
                        num_splits: splits,
                    },
                )?;
        }
    }
    Ok(())
}

fn launch_128(
    runtime: &CudaRuntime,
    fixture: &Fixture,
    splits: usize,
    partials: &mut crate::tensor::Tensor<f32>,
    output: &mut crate::tensor::Tensor<bf16>,
) -> Result<()> {
    unsafe {
        if splits == 1 {
            runtime.kernels().attention_cta128().launch_ragged_lfm2_bf16(
                runtime.stream(),
                FastRaggedAttentionLaunch {
                    page_size: 16,
                    query: fixture.query.storage(),
                    key_cache: fixture.arena.key().storage(),
                    value_cache: fixture.arena.value().storage(),
                    block_tables: fixture.block_tables.storage(),
                    request_slots: fixture.request_slots.storage(),
                    position_ids: fixture.positions.storage(),
                    output: output.storage_mut(),
                    num_tokens: fixture.query.dims()[0],
                    num_pages: fixture.arena.num_pages(),
                    block_table_stride: fixture.pages_per_request,
                },
            )?;
        } else {
            runtime
                .kernels()
                .attention_cta128()
                .launch_splitk_ragged_lfm2_bf16(
                    runtime.stream(),
                    SplitKRaggedAttentionLaunch {
                        page_size: 16,
                        query: fixture.query.storage(),
                        key_cache: fixture.arena.key().storage(),
                        value_cache: fixture.arena.value().storage(),
                        block_tables: fixture.block_tables.storage(),
                        request_slots: fixture.request_slots.storage(),
                        position_ids: fixture.positions.storage(),
                        partials: partials.storage_mut(),
                        output: output.storage_mut(),
                        num_tokens: fixture.query.dims()[0],
                        num_pages: fixture.arena.num_pages(),
                        block_table_stride: fixture.pages_per_request,
                        num_splits: splits,
                    },
                )?;
        }
    }
    Ok(())
}

#[test]
fn cta128_attention_matches_cta256_nonzero_ps16() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let fixture = fixture(&runtime, 2, 65, true)?;

    for splits in [1usize, 2, 4, 8] {
        let mut reference = runtime.alloc_bf16(Shape::new([2, 32, 64]))?;
        let mut candidate = runtime.alloc_bf16(Shape::new([2, 32, 64]))?;
        let mut reference_partials = runtime.alloc_uninit::<f32>(Shape::new([
            splitk_workspace_elements(2)?,
        ]))?;
        let mut candidate_partials = runtime.alloc_uninit::<f32>(Shape::new([
            splitk_workspace_elements(2)?,
        ]))?;

        launch_256(
            &runtime,
            &fixture,
            splits,
            &mut reference_partials,
            &mut reference,
        )?;
        launch_128(
            &runtime,
            &fixture,
            splits,
            &mut candidate_partials,
            &mut candidate,
        )?;
        runtime.synchronize()?;
        assert_eq_bf16(
            &readback(&runtime, &candidate)?,
            &readback(&runtime, &reference)?,
        );
    }
    Ok(())
}

fn bench_case(runtime: &CudaRuntime, batch: usize, context: usize, config: BenchConfig) -> Result<()> {
    let fixture = fixture(runtime, batch, context, false)?;
    let splits = splitk_decode_splits(batch, context, 16);
    let mut reference = runtime.alloc_bf16(Shape::new([batch, 32, 64]))?;
    let mut candidate = runtime.alloc_bf16(Shape::new([batch, 32, 64]))?;
    let mut reference_partials = runtime.alloc_uninit::<f32>(Shape::new([
        splitk_workspace_elements(batch)?,
    ]))?;
    let mut candidate_partials = runtime.alloc_uninit::<f32>(Shape::new([
        splitk_workspace_elements(batch)?,
    ]))?;

    let paired = benchmark_gpu_paired(
        runtime.context(),
        runtime.stream(),
        config,
        || launch_256(runtime, &fixture, splits, &mut reference_partials, &mut reference),
        || launch_128(runtime, &fixture, splits, &mut candidate_partials, &mut candidate),
    )?;

    println!(
        "cta_geometry page_size=16 batch={batch} context={context} splits={splits} cta256_mean={:.3}us cta256_p50={:.3}us cta256_p95={:.3}us cta128_mean={:.3}us cta128_p50={:.3}us cta128_p95={:.3}us speedup_mean={:.4}x speedup_p50={:.4}x speedup_p95={:.4}x",
        paired.reference.mean_us,
        paired.reference.p50_us,
        paired.reference.p95_us,
        paired.candidate.mean_us,
        paired.candidate.p50_us,
        paired.candidate.p95_us,
        paired.speedup_mean,
        paired.speedup_p50,
        paired.speedup_p95,
    );
    Ok(())
}

#[test]
#[ignore = "GPU benchmark"]
fn bench_cta128_vs_cta256_ps16() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let config = BenchConfig {
        warmup: 8,
        batches: 20,
        iterations_per_batch: 10,
    };

    for (batch, contexts) in [
        (1usize, &[128usize, 512, 2048, 8192][..]),
        (2usize, &[128usize, 512, 2048, 8192][..]),
        (4usize, &[128usize, 512, 2048, 8192][..]),
        (8usize, &[128usize, 512, 2048, 8192][..]),
        (16usize, &[128usize, 512, 2048, 8192][..]),
        (32usize, &[128usize, 512, 2048][..]),
        (64usize, &[128usize, 512, 2048][..]),
    ] {
        for &context in contexts {
            bench_case(&runtime, batch, context, config)?;
        }
    }
    Ok(())
}
