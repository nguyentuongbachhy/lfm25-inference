use std::mem::size_of;

use anyhow::{Context as _, Result, ensure};
use half::bf16;

use crate::{
    cache::{KvPageSize, PagedKvCache},
    cuda::{
        CudaRuntime, Fp8KvAttentionLaunch, Fp8KvQuantizeLaunch, PagedAttentionLaunch,
        benchmark::{BenchConfig, benchmark_gpu_paired},
    },
    tensor::Shape,
};

const NUM_Q_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const PAGE_SIZE: usize = 16;

#[derive(Debug, Clone, Copy)]
struct QualityMetrics {
    nrmse: f64,
    cosine: f64,
    max_abs_error: f64,
    non_finite: usize,
}

fn deterministic_bf16(index: usize, salt: u32) -> bf16 {
    let mut value = (index as u32).wrapping_mul(747_796_405).wrapping_add(salt);
    value ^= value >> 16;
    value = value.wrapping_mul(2_246_822_519);
    value ^= value >> 13;
    let unit = value as f32 / u32::MAX as f32;
    bf16::from_f32((unit * 2.0 - 1.0) * 0.75)
}

fn quality_metrics(reference: &[bf16], candidate: &[bf16]) -> QualityMetrics {
    assert_eq!(reference.len(), candidate.len());
    let mut squared_error = 0.0f64;
    let mut reference_energy = 0.0f64;
    let mut candidate_energy = 0.0f64;
    let mut dot = 0.0f64;
    let mut max_abs_error = 0.0f64;
    let mut non_finite = 0usize;

    for (&reference, &candidate) in reference.iter().zip(candidate) {
        let reference = f64::from(reference.to_f32());
        let candidate = f64::from(candidate.to_f32());
        if !reference.is_finite() || !candidate.is_finite() {
            non_finite += 1;
            continue;
        }
        let error = candidate - reference;
        squared_error += error * error;
        reference_energy += reference * reference;
        candidate_energy += candidate * candidate;
        dot += reference * candidate;
        max_abs_error = max_abs_error.max(error.abs());
    }

    let nrmse = if reference_energy > 0.0 {
        (squared_error / reference_energy).sqrt()
    } else if squared_error == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    let cosine_denominator = (reference_energy * candidate_energy).sqrt();
    let cosine = if cosine_denominator > 0.0 {
        dot / cosine_denominator
    } else if reference_energy == 0.0 && candidate_energy == 0.0 {
        1.0
    } else {
        0.0
    };

    QualityMetrics {
        nrmse,
        cosine,
        max_abs_error,
        non_finite,
    }
}

fn run_case(runtime: &CudaRuntime, context: usize, benchmark: Option<BenchConfig>) -> Result<()> {
    ensure!(context > 0, "FP8 KV test context must be positive");
    let page_size = KvPageSize::P16;
    let num_pages = context.div_ceil(PAGE_SIZE);
    let cache_elements = num_pages
        .checked_mul(NUM_KV_HEADS * PAGE_SIZE * HEAD_DIM)
        .context("FP8 KV test cache size overflow")?;

    let key_host = (0..context * NUM_KV_HEADS * HEAD_DIM)
        .map(|index| deterministic_bf16(index, 0x9e37_79b9))
        .collect::<Vec<_>>();
    let value_host = (0..context * NUM_KV_HEADS * HEAD_DIM)
        .map(|index| deterministic_bf16(index, 0x243f_6a88))
        .collect::<Vec<_>>();
    let key = runtime.upload(&key_host, Shape::new([context, NUM_KV_HEADS, HEAD_DIM]))?;
    let value = runtime.upload(&value_host, Shape::new([context, NUM_KV_HEADS, HEAD_DIM]))?;
    let slots_host = (0..context)
        .map(i64::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let slots = runtime.upload(&slots_host, Shape::new([context]))?;
    let mut cache = PagedKvCache::new(runtime, context, page_size)?;
    cache.write_lfm2(runtime, &key, &value, &slots)?;

    let query_host = (0..NUM_Q_HEADS * HEAD_DIM)
        .map(|index| deterministic_bf16(index, 0xb7e1_5163))
        .collect::<Vec<_>>();
    let query = runtime.upload(&query_host, Shape::new([1, NUM_Q_HEADS, HEAD_DIM]))?;
    let position = runtime.upload(&[u32::try_from(context - 1)?], Shape::new([1]))?;

    let mut key_fp8 = runtime.zeros::<u8>(Shape::new([cache_elements]))?;
    let mut value_fp8 = runtime.zeros::<u8>(Shape::new([cache_elements]))?;
    let mut key_scales = runtime.zeros::<f32>(Shape::new([num_pages, NUM_KV_HEADS]))?;
    let mut value_scales = runtime.zeros::<f32>(Shape::new([num_pages, NUM_KV_HEADS]))?;

    unsafe {
        runtime.kernels().attention_fp8_kv().launch_quantize_ps16(
            runtime.stream(),
            Fp8KvQuantizeLaunch {
                key_cache: cache.key().storage(),
                value_cache: cache.value().storage(),
                key_fp8: key_fp8.storage_mut(),
                value_fp8: value_fp8.storage_mut(),
                key_scales: key_scales.storage_mut(),
                value_scales: value_scales.storage_mut(),
                num_pages,
            },
        )?;
    }

    let mut reference_output = runtime.alloc_bf16(Shape::new([1, NUM_Q_HEADS, HEAD_DIM]))?;
    let mut candidate_output = runtime.alloc_bf16(Shape::new([1, NUM_Q_HEADS, HEAD_DIM]))?;

    unsafe {
        runtime.kernels().attention_async_fast().launch_lfm2_bf16(
            runtime.stream(),
            PagedAttentionLaunch {
                page_size: PAGE_SIZE,
                query: query.storage(),
                key_cache: cache.key().storage(),
                value_cache: cache.value().storage(),
                block_table: cache.block_table().storage(),
                position_ids: position.storage(),
                output: reference_output.storage_mut(),
                num_tokens: 1,
                num_pages,
            },
        )?;
        runtime.kernels().attention_fp8_kv().launch_attention_ps16(
            runtime.stream(),
            Fp8KvAttentionLaunch {
                query: query.storage(),
                key_cache: key_fp8.storage(),
                value_cache: value_fp8.storage(),
                key_scales: key_scales.storage(),
                value_scales: value_scales.storage(),
                block_table: cache.block_table().storage(),
                position_ids: position.storage(),
                output: candidate_output.storage_mut(),
                num_tokens: 1,
                num_pages,
            },
        )?;
    }
    runtime.stream().synchronize()?;

    let reference_host = runtime.download(&reference_output)?;
    let candidate_host = runtime.download(&candidate_output)?;
    let quality = quality_metrics(&reference_host, &candidate_host);
    println!(
        "fp8_kv_quality context={} nrmse={:.6} cosine={:.8} max_abs={:.6} non_finite={}",
        context, quality.nrmse, quality.cosine, quality.max_abs_error, quality.non_finite,
    );
    ensure!(quality.non_finite == 0, "FP8 KV produced non-finite output");
    ensure!(
        quality.nrmse <= 0.05,
        "FP8 KV primitive NRMSE gate failed: {:.6}",
        quality.nrmse
    );
    ensure!(
        quality.cosine >= 0.999,
        "FP8 KV primitive cosine gate failed: {:.8}",
        quality.cosine
    );

    if let Some(config) = benchmark {
        let paired = benchmark_gpu_paired(
            runtime.context(),
            runtime.stream(),
            config,
            || {
                unsafe {
                    runtime.kernels().attention_async_fast().launch_lfm2_bf16(
                        runtime.stream(),
                        PagedAttentionLaunch {
                            page_size: PAGE_SIZE,
                            query: query.storage(),
                            key_cache: cache.key().storage(),
                            value_cache: cache.value().storage(),
                            block_table: cache.block_table().storage(),
                            position_ids: position.storage(),
                            output: reference_output.storage_mut(),
                            num_tokens: 1,
                            num_pages,
                        },
                    )?;
                }
                Ok(())
            },
            || {
                unsafe {
                    runtime.kernels().attention_fp8_kv().launch_attention_ps16(
                        runtime.stream(),
                        Fp8KvAttentionLaunch {
                            query: query.storage(),
                            key_cache: key_fp8.storage(),
                            value_cache: value_fp8.storage(),
                            key_scales: key_scales.storage(),
                            value_scales: value_scales.storage(),
                            block_table: cache.block_table().storage(),
                            position_ids: position.storage(),
                            output: candidate_output.storage_mut(),
                            num_tokens: 1,
                            num_pages,
                        },
                    )?;
                }
                Ok(())
            },
        )?;
        let bf16_payload_bytes = context
            .checked_mul(NUM_KV_HEADS * HEAD_DIM * 2 * 2)
            .context("BF16 KV payload size overflow")?;
        let fp8_payload_bytes = context
            .checked_mul(NUM_KV_HEADS * HEAD_DIM * 2)
            .and_then(|bytes| bytes.checked_add(num_pages * NUM_KV_HEADS * 2 * size_of::<f32>()))
            .context("FP8 KV payload size overflow")?;
        println!(
            "fp8_kv_bench context={} bf16_mean={:.3}us bf16_p50={:.3}us bf16_p95={:.3}us fp8_mean={:.3}us fp8_p50={:.3}us fp8_p95={:.3}us speedup_mean={:.4}x speedup_p50={:.4}x speedup_p95={:.4}x speedup_min={:.4}x speedup_max={:.4}x bf16_payload_bytes={} fp8_payload_bytes={} payload_ratio={:.4}",
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
            bf16_payload_bytes,
            fp8_payload_bytes,
            fp8_payload_bytes as f64 / bf16_payload_bytes as f64,
        );
    }

    Ok(())
}

#[test]
#[ignore = "GPU research quality test"]
fn fp8_kv_attention_quality_smoke_ps16() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    for context in [128usize, 512, 2048, 8192] {
        run_case(&runtime, context, None)?;
    }
    Ok(())
}

#[test]
#[ignore = "GPU research benchmark"]
fn bench_fp8_kv_attention_ps16() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let config = BenchConfig {
        warmup: 10,
        batches: 30,
        iterations_per_batch: 20,
    };
    for context in [128usize, 512, 2048, 8192] {
        run_case(&runtime, context, Some(config))?;
    }
    Ok(())
}
