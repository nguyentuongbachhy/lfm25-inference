use anyhow::{Context as _, Result, ensure};
use half::bf16;

use super::{
    CudaRuntime,
    benchmark::{BenchConfig, benchmark_gpu_paired},
};
use crate::{
    ops::{self, prefill_dispatch::ScopedFlashPrefillOverride},
    tensor::Shape,
};

const Q_HEADS: usize = 32;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const TOKEN_COUNTS: &[usize] = &[512, 2048, 8192];
const ATOL: f64 = 0.035;
const RTOL: f64 = 0.025;

#[derive(Debug)]
struct NumericalMetrics {
    max_abs: f64,
    nrmse: f64,
    cosine: f64,
    non_finite: usize,
    within_tolerance: bool,
}

fn deterministic_bf16(elements: usize, seed: usize) -> Vec<bf16> {
    (0..elements)
        .map(|index| {
            let mixed = index
                .wrapping_mul(1_103_515_245)
                .wrapping_add(seed.wrapping_mul(12_345));
            let bucket = (mixed >> 8) & 255;
            bf16::from_f32((bucket as f32 - 127.5) / 512.0)
        })
        .collect()
}

fn numerical_metrics(reference: &[bf16], candidate: &[bf16]) -> NumericalMetrics {
    assert_eq!(reference.len(), candidate.len());
    let mut max_abs = 0.0f64;
    let mut squared_error = 0.0f64;
    let mut squared_reference = 0.0f64;
    let mut squared_candidate = 0.0f64;
    let mut dot = 0.0f64;
    let mut non_finite = 0usize;
    let mut within_tolerance = true;

    for (&expected, &observed) in reference.iter().zip(candidate) {
        let expected = f64::from(expected.to_f32());
        let observed = f64::from(observed.to_f32());
        if !expected.is_finite() || !observed.is_finite() {
            non_finite += 1;
            within_tolerance = false;
            continue;
        }
        let difference = observed - expected;
        let absolute = difference.abs();
        max_abs = max_abs.max(absolute);
        squared_error += difference * difference;
        squared_reference += expected * expected;
        squared_candidate += observed * observed;
        dot += expected * observed;
        if absolute > ATOL + RTOL * expected.abs() {
            within_tolerance = false;
        }
    }

    let nrmse = if squared_reference > 0.0 {
        (squared_error / squared_reference).sqrt()
    } else if squared_error == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    let cosine = if squared_reference > 0.0 && squared_candidate > 0.0 {
        dot / (squared_reference * squared_candidate).sqrt()
    } else if squared_reference == squared_candidate {
        1.0
    } else {
        0.0
    };

    NumericalMetrics {
        max_abs,
        nrmse,
        cosine,
        non_finite,
        within_tolerance,
    }
}

#[test]
fn test_prefill_flash_correctness() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;

    for &num_tokens in &[1, 7, 16, 64, 128, 512] {
        let q_elements = num_tokens * Q_HEADS * HEAD_DIM;
        let kv_elements = num_tokens * KV_HEADS * HEAD_DIM;

        let query = runtime.upload(
            &deterministic_bf16(q_elements, 17),
            Shape::new([num_tokens, Q_HEADS, HEAD_DIM]),
        )?;
        let key = runtime.upload(
            &deterministic_bf16(kv_elements, 29),
            Shape::new([num_tokens, KV_HEADS, HEAD_DIM]),
        )?;
        let value = runtime.upload(
            &deterministic_bf16(kv_elements, 43),
            Shape::new([num_tokens, KV_HEADS, HEAD_DIM]),
        )?;
        let mut reference = runtime.alloc_bf16(Shape::new([num_tokens, Q_HEADS, HEAD_DIM]))?;
        let mut candidate = runtime.alloc_bf16(Shape::new([num_tokens, Q_HEADS, HEAD_DIM]))?;

        unsafe {
            runtime.kernels().attention().launch_prefill_lfm2_bf16(
                runtime.stream(),
                query.storage(),
                key.storage(),
                value.storage(),
                reference.storage_mut(),
                num_tokens,
            )?;
            runtime
                .kernels()
                .attention()
                .launch_prefill_flash_lfm2_bf16(
                    runtime.stream(),
                    query.storage(),
                    key.storage(),
                    value.storage(),
                    candidate.storage_mut(),
                    num_tokens,
                )?;
        }
        runtime.synchronize()?;
        let reference_host = runtime.download(&reference)?;
        let candidate_host = runtime.download(&candidate)?;
        let metrics = numerical_metrics(&reference_host, &candidate_host);
        ensure!(
            metrics.non_finite == 0,
            "flash prefill produced non-finite output at N={num_tokens}"
        );
        ensure!(
            metrics.within_tolerance,
            "flash prefill exceeds tolerance at N={num_tokens}: {metrics:?}"
        );
    }
    Ok(())
}

#[test]
fn test_prefill_attention_op_dispatch_matches_reference() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let num_tokens = 128;
    let q_elements = num_tokens * Q_HEADS * HEAD_DIM;
    let kv_elements = num_tokens * KV_HEADS * HEAD_DIM;

    let query = runtime.upload(
        &deterministic_bf16(q_elements, 17),
        Shape::new([num_tokens, Q_HEADS, HEAD_DIM]),
    )?;
    let key = runtime.upload(
        &deterministic_bf16(kv_elements, 29),
        Shape::new([num_tokens, KV_HEADS, HEAD_DIM]),
    )?;
    let value = runtime.upload(
        &deterministic_bf16(kv_elements, 43),
        Shape::new([num_tokens, KV_HEADS, HEAD_DIM]),
    )?;

    // Reference (flash prefill disabled)
    let ref_out = {
        let _guard = ScopedFlashPrefillOverride::new(false);
        ops::prefill_attention_lfm2_bf16(&runtime, &query, &key, &value)?
    };

    // Candidate (flash prefill enabled)
    let cand_out = {
        let _guard = ScopedFlashPrefillOverride::new(true);
        ops::prefill_attention_lfm2_bf16(&runtime, &query, &key, &value)?
    };

    let ref_host = runtime.download(&ref_out)?;
    let cand_host = runtime.download(&cand_out)?;
    let metrics = numerical_metrics(&ref_host, &cand_host);
    ensure!(
        metrics.within_tolerance && metrics.non_finite == 0,
        "prefill op dispatch mismatch: {metrics:?}"
    );
    Ok(())
}

#[test]
#[ignore = "GPU benchmark: Tensor Core FlashAttention contiguous prefill"]
fn bench_prefill_attention_flash_bf16() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;

    for &num_tokens in TOKEN_COUNTS {
        let q_elements = num_tokens
            .checked_mul(Q_HEADS * HEAD_DIM)
            .context("flash query element count overflow")?;
        let kv_elements = num_tokens
            .checked_mul(KV_HEADS * HEAD_DIM)
            .context("flash KV element count overflow")?;

        let query = runtime.upload(
            &deterministic_bf16(q_elements, 17),
            Shape::new([num_tokens, Q_HEADS, HEAD_DIM]),
        )?;
        let key = runtime.upload(
            &deterministic_bf16(kv_elements, 29),
            Shape::new([num_tokens, KV_HEADS, HEAD_DIM]),
        )?;
        let value = runtime.upload(
            &deterministic_bf16(kv_elements, 43),
            Shape::new([num_tokens, KV_HEADS, HEAD_DIM]),
        )?;
        let mut reference = runtime.alloc_bf16(Shape::new([num_tokens, Q_HEADS, HEAD_DIM]))?;
        let mut candidate = runtime.alloc_bf16(Shape::new([num_tokens, Q_HEADS, HEAD_DIM]))?;

        unsafe {
            runtime.kernels().attention().launch_prefill_lfm2_bf16(
                runtime.stream(),
                query.storage(),
                key.storage(),
                value.storage(),
                reference.storage_mut(),
                num_tokens,
            )?;
            runtime
                .kernels()
                .attention()
                .launch_prefill_flash_lfm2_bf16(
                    runtime.stream(),
                    query.storage(),
                    key.storage(),
                    value.storage(),
                    candidate.storage_mut(),
                    num_tokens,
                )?;
        }
        runtime.synchronize()?;
        let reference_host = runtime.download(&reference)?;
        let candidate_host = runtime.download(&candidate)?;
        let metrics = numerical_metrics(&reference_host, &candidate_host);
        ensure!(
            metrics.non_finite == 0,
            "flash prefill produced non-finite output at N={num_tokens}"
        );
        ensure!(
            metrics.within_tolerance,
            "flash prefill exceeds tolerance at N={num_tokens}: {metrics:?}"
        );

        let stats = benchmark_gpu_paired(
            runtime.context(),
            runtime.stream(),
            BenchConfig {
                warmup: 3,
                batches: 10,
                iterations_per_batch: 1,
            },
            || unsafe {
                runtime.kernels().attention().launch_prefill_lfm2_bf16(
                    runtime.stream(),
                    query.storage(),
                    key.storage(),
                    value.storage(),
                    reference.storage_mut(),
                    num_tokens,
                )
            },
            || unsafe {
                runtime
                    .kernels()
                    .attention()
                    .launch_prefill_flash_lfm2_bf16(
                        runtime.stream(),
                        query.storage(),
                        key.storage(),
                        value.storage(),
                        candidate.storage_mut(),
                        num_tokens,
                    )
            },
        )?;

        println!(
            "prefill_flash N={} precise_mean_us={:.3} fast_mean_us={:.3} speedup_mean={:.4}x precise_p50_us={:.3} fast_p50_us={:.3} speedup_p50={:.4}x precise_p95_us={:.3} fast_p95_us={:.3} speedup_p95={:.4}x speedup_min={:.4}x speedup_max={:.4}x max_abs={:.8} nrmse={:.8} cosine={:.8} non_finite={} within_tolerance={}",
            num_tokens,
            stats.reference.mean_us,
            stats.candidate.mean_us,
            stats.speedup_mean,
            stats.reference.p50_us,
            stats.candidate.p50_us,
            stats.speedup_p50,
            stats.reference.p95_us,
            stats.candidate.p95_us,
            stats.speedup_p95,
            stats.speedup_min,
            stats.speedup_max,
            metrics.max_abs,
            metrics.nrmse,
            metrics.cosine,
            metrics.non_finite,
            metrics.within_tolerance,
        );
    }

    Ok(())
}
