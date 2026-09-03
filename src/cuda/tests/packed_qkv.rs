use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cuda::{
        CudaRuntime, QkvUnpackLaunch,
        benchmark::{BenchConfig, benchmark_gpu_paired},
        testing::readback,
    },
    tensor::{Shape, Tensor},
};

const K: usize = 2048;
const Q: usize = 2048;
const KV: usize = 512;
const PACKED: usize = Q + 2 * KV;

#[derive(Debug, Clone, Copy)]
struct Quality {
    nrmse: f64,
    cosine: f64,
    max_abs: f64,
    non_finite: usize,
}

fn values(elements: usize, mul: usize, modulus: usize, center: f32, scale: f32) -> Vec<bf16> {
    (0..elements)
        .map(|index| {
            let raw = ((index.wrapping_mul(mul) % modulus) as f32 - center) / scale;
            bf16::from_f32(raw)
        })
        .collect()
}

fn quality(reference: &[bf16], candidate: &[bf16]) -> Quality {
    assert_eq!(reference.len(), candidate.len());
    let mut squared_error = 0.0f64;
    let mut squared_reference = 0.0f64;
    let mut reference_norm = 0.0f64;
    let mut candidate_norm = 0.0f64;
    let mut dot = 0.0f64;
    let mut max_abs = 0.0f64;
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
        squared_reference += reference * reference;
        reference_norm += reference * reference;
        candidate_norm += candidate * candidate;
        dot += reference * candidate;
        max_abs = max_abs.max(error.abs());
    }

    Quality {
        nrmse: (squared_error / squared_reference.max(1.0e-30)).sqrt(),
        cosine: dot / (reference_norm.sqrt() * candidate_norm.sqrt()).max(1.0e-30),
        max_abs,
        non_finite,
    }
}

fn merged_quality(parts: &[Quality]) -> Quality {
    Quality {
        nrmse: parts.iter().map(|metric| metric.nrmse).fold(0.0, f64::max),
        cosine: parts
            .iter()
            .map(|metric| metric.cosine)
            .fold(1.0, f64::min),
        max_abs: parts
            .iter()
            .map(|metric| metric.max_abs)
            .fold(0.0, f64::max),
        non_finite: parts.iter().map(|metric| metric.non_finite).sum(),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_direct(
    runtime: &CudaRuntime,
    input: &Tensor<bf16>,
    q_weight: &Tensor<bf16>,
    k_weight: &Tensor<bf16>,
    v_weight: &Tensor<bf16>,
    q: &mut Tensor<bf16>,
    k: &mut Tensor<bf16>,
    v: &mut Tensor<bf16>,
    m: usize,
) -> Result<()> {
    unsafe {
        runtime.blaslt().linear_bf16(
            input.storage(),
            q_weight.storage(),
            q.storage_mut(),
            m,
            Q,
            K,
        )?;
        runtime.blaslt().linear_bf16(
            input.storage(),
            k_weight.storage(),
            k.storage_mut(),
            m,
            KV,
            K,
        )?;
        runtime.blaslt().linear_bf16(
            input.storage(),
            v_weight.storage(),
            v.storage_mut(),
            m,
            KV,
            K,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_candidate(
    runtime: &CudaRuntime,
    input: &Tensor<bf16>,
    packed_weight: &Tensor<bf16>,
    packed: &mut Tensor<bf16>,
    q: &mut Tensor<bf16>,
    k: &mut Tensor<bf16>,
    v: &mut Tensor<bf16>,
    m: usize,
) -> Result<()> {
    unsafe {
        runtime.blaslt().linear_bf16(
            input.storage(),
            packed_weight.storage(),
            packed.storage_mut(),
            m,
            PACKED,
            K,
        )?;
        runtime.kernels().qkv_unpack().launch_bf16(
            runtime.stream(),
            QkvUnpackLaunch {
                packed: packed.storage(),
                query: q.storage_mut(),
                key: k.storage_mut(),
                value: v.storage_mut(),
                num_tokens: m,
            },
        )?;
    }
    Ok(())
}

#[test]
#[ignore = "GPU benchmark"]
fn bench_packed_qkv_bf16() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let q_weight = runtime.upload(
        &values(Q * K, 17, 257, 128.0, 256.0),
        Shape::new([Q, K]),
    )?;
    let k_weight = runtime.upload(
        &values(KV * K, 19, 251, 125.0, 256.0),
        Shape::new([KV, K]),
    )?;
    let v_weight = runtime.upload(
        &values(KV * K, 23, 241, 120.0, 256.0),
        Shape::new([KV, K]),
    )?;
    let qk_weight = runtime.pack_rows_bf16(&q_weight, &k_weight)?;
    let packed_weight = runtime.pack_rows_bf16(&qk_weight, &v_weight)?;
    ensure!(
        packed_weight.dims() == [PACKED, K],
        "packed QKV weight shape mismatch"
    );

    let config = BenchConfig {
        warmup: 20,
        batches: 40,
        iterations_per_batch: 20,
    };

    for m in [1usize, 8, 16] {
        runtime.blaslt().prepare_linear_bf16(m, Q, K)?;
        runtime.blaslt().prepare_linear_bf16(m, KV, K)?;
        runtime.blaslt().prepare_linear_bf16(m, PACKED, K)?;

        let input = runtime.upload(
            &values(m * K, 29, 233, 116.0, 128.0),
            Shape::new([m, K]),
        )?;
        let mut direct_q = runtime.alloc_uninit::<bf16>(Shape::new([m, Q]))?;
        let mut direct_k = runtime.alloc_uninit::<bf16>(Shape::new([m, KV]))?;
        let mut direct_v = runtime.alloc_uninit::<bf16>(Shape::new([m, KV]))?;
        let mut packed = runtime.alloc_uninit::<bf16>(Shape::new([m, PACKED]))?;
        let mut candidate_q = runtime.alloc_uninit::<bf16>(Shape::new([m, Q]))?;
        let mut candidate_k = runtime.alloc_uninit::<bf16>(Shape::new([m, KV]))?;
        let mut candidate_v = runtime.alloc_uninit::<bf16>(Shape::new([m, KV]))?;

        run_direct(
            &runtime,
            &input,
            &q_weight,
            &k_weight,
            &v_weight,
            &mut direct_q,
            &mut direct_k,
            &mut direct_v,
            m,
        )?;
        run_candidate(
            &runtime,
            &input,
            &packed_weight,
            &mut packed,
            &mut candidate_q,
            &mut candidate_k,
            &mut candidate_v,
            m,
        )?;
        runtime.synchronize()?;
        let metrics = merged_quality(&[
            quality(
                &readback(&runtime, &direct_q)?,
                &readback(&runtime, &candidate_q)?,
            ),
            quality(
                &readback(&runtime, &direct_k)?,
                &readback(&runtime, &candidate_k)?,
            ),
            quality(
                &readback(&runtime, &direct_v)?,
                &readback(&runtime, &candidate_v)?,
            ),
        ]);
        ensure!(
            metrics.non_finite == 0,
            "packed QKV produced non-finite values"
        );
        ensure!(
            metrics.nrmse <= 0.01,
            "packed QKV NRMSE gate failed: {metrics:?}"
        );
        ensure!(
            metrics.cosine >= 0.9999,
            "packed QKV cosine gate failed: {metrics:?}"
        );

        let stats = benchmark_gpu_paired(
            runtime.context(),
            runtime.stream(),
            config,
            || {
                run_direct(
                    &runtime,
                    &input,
                    &q_weight,
                    &k_weight,
                    &v_weight,
                    &mut direct_q,
                    &mut direct_k,
                    &mut direct_v,
                    m,
                )
            },
            || {
                run_candidate(
                    &runtime,
                    &input,
                    &packed_weight,
                    &mut packed,
                    &mut candidate_q,
                    &mut candidate_k,
                    &mut candidate_v,
                    m,
                )
            },
        )?;
        println!(
            "packed_qkv_bench M={} nrmse={:.8} cosine={:.8} max_abs={:.6} non_finite={} direct_mean_us={:.3} direct_p50_us={:.3} direct_p95_us={:.3} candidate_mean_us={:.3} candidate_p50_us={:.3} candidate_p95_us={:.3} speedup_mean={:.4}x speedup_p50={:.4}x speedup_p95={:.4}x speedup_min={:.4}x speedup_max={:.4}x",
            m,
            metrics.nrmse,
            metrics.cosine,
            metrics.max_abs,
            metrics.non_finite,
            stats.reference.mean_us,
            stats.reference.p50_us,
            stats.reference.p95_us,
            stats.candidate.mean_us,
            stats.candidate.p50_us,
            stats.candidate.p95_us,
            stats.speedup_mean,
            stats.speedup_p50,
            stats.speedup_p95,
            stats.speedup_min,
            stats.speedup_max,
        );
    }
    Ok(())
}
