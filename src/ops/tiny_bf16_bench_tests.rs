use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cuda::{
        CudaRuntime, TINY_BF16_MAX_M,
        benchmark::{BenchConfig, benchmark_gpu_paired},
    },
    tensor::{Shape, Tensor},
};

const ROTATING_WEIGHT_COUNT: usize = 4;

fn tiny_bf16_nt_into(
    runtime: &CudaRuntime,
    input: &Tensor<bf16>,
    weight: &Tensor<bf16>,
    output: &mut Tensor<bf16>,
) -> Result<()> {
    ensure!(input.rank() == 2, "tiny BF16 benchmark input must be rank 2");
    ensure!(weight.rank() == 2, "tiny BF16 benchmark weight must be rank 2");
    let m = input.dims()[0];
    let k = input.dims()[1];
    let n = weight.dims()[0];
    ensure!(m > 0 && m <= TINY_BF16_MAX_M, "tiny BF16 benchmark M out of range");
    ensure!(weight.dims()[1] == k, "tiny BF16 benchmark K mismatch");
    output.set_logical_shape(Shape::new([m, n]))?;
    unsafe {
        runtime.kernels().tiny_bf16().launch_nt_m8(
            runtime.stream(),
            input.storage(),
            weight.storage(),
            output.storage_mut(),
            m,
            n,
            k,
        )?;
    }
    Ok(())
}

fn deterministic_values(elements: usize, multiplier: usize, modulus: usize) -> Vec<bf16> {
    (0..elements)
        .map(|index| {
            let centered = ((index.wrapping_mul(multiplier) + index / 97) % modulus) as f32
                - (modulus as f32 - 1.0) * 0.5;
            bf16::from_f32(centered / modulus as f32)
        })
        .collect()
}

fn output_metrics(reference: &[bf16], candidate: &[bf16]) -> (f64, f64, f64) {
    let mut squared_error = 0.0f64;
    let mut reference_energy = 0.0f64;
    let mut candidate_energy = 0.0f64;
    let mut dot = 0.0f64;
    let mut max_abs = 0.0f64;
    for (&reference, &candidate) in reference.iter().zip(candidate) {
        let reference = f64::from(reference.to_f32());
        let candidate = f64::from(candidate.to_f32());
        let delta = reference - candidate;
        squared_error += delta * delta;
        reference_energy += reference * reference;
        candidate_energy += candidate * candidate;
        dot += reference * candidate;
        max_abs = max_abs.max(delta.abs());
    }
    let rel_l2 = (squared_error / reference_energy.max(f64::MIN_POSITIVE)).sqrt();
    let cosine = dot
        / (reference_energy * candidate_energy)
            .sqrt()
            .max(f64::MIN_POSITIVE);
    (rel_l2, cosine, max_abs)
}

#[test]
fn tiny_bf16_matches_cublaslt_on_small_shape() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let m = 2usize;
    let n = 16usize;
    let k = 64usize;
    let input = runtime.upload(
        &deterministic_values(m * k, 17, 31),
        Shape::new([m, k]),
    )?;
    let weight = runtime.upload(
        &deterministic_values(n * k, 29, 37),
        Shape::new([n, k]),
    )?;
    let mut reference = runtime.alloc_uninit::<bf16>(Shape::new([m, n]))?;
    let mut candidate = runtime.alloc_uninit::<bf16>(Shape::new([m, n]))?;

    unsafe {
        runtime.blaslt().linear_bf16(
            input.storage(),
            weight.storage(),
            reference.storage_mut(),
            m,
            n,
            k,
        )?;
    }
    tiny_bf16_nt_into(&runtime, &input, &weight, &mut candidate)?;
    runtime.synchronize()?;

    let reference = runtime.download(&reference)?;
    let candidate = runtime.download(&candidate)?;
    let (rel_l2, cosine, max_abs) = output_metrics(&reference, &candidate);
    ensure!(rel_l2 <= 0.01, "tiny BF16 small-shape rel_l2 too large: {rel_l2}");
    ensure!(cosine >= 0.9999, "tiny BF16 small-shape cosine too low: {cosine}");
    ensure!(max_abs.is_finite(), "tiny BF16 small-shape max_abs is not finite");
    Ok(())
}

#[test]
#[ignore = "GPU microbenchmark for real decode GEMM shapes"]
fn bench_tiny_bf16_decode_shapes() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let bench = BenchConfig {
        warmup: 12,
        batches: 30,
        iterations_per_batch: 10,
    };

    for (site, n, k) in [
        ("down", 2_048usize, 8_192usize),
        ("gate_up", 16_384usize, 2_048usize),
    ] {
        let weight = runtime.upload(
            &deterministic_values(n * k, 29, 257),
            Shape::new([n, k]),
        )?;

        for m in [1usize, 2, 4, 8] {
            let input = runtime.upload(
                &deterministic_values(m * k, 17 + m, 251),
                Shape::new([m, k]),
            )?;
            let mut reference = runtime.alloc_uninit::<bf16>(Shape::new([m, n]))?;
            let mut candidate = runtime.alloc_uninit::<bf16>(Shape::new([m, n]))?;

            unsafe {
                runtime.blaslt().linear_bf16(
                    input.storage(),
                    weight.storage(),
                    reference.storage_mut(),
                    m,
                    n,
                    k,
                )?;
            }
            tiny_bf16_nt_into(&runtime, &input, &weight, &mut candidate)?;
            runtime.synchronize()?;

            let reference_host = runtime.download(&reference)?;
            let candidate_host = runtime.download(&candidate)?;
            let (rel_l2, cosine, max_abs) = output_metrics(&reference_host, &candidate_host);

            let stats = benchmark_gpu_paired(
                runtime.context(),
                runtime.stream(),
                bench,
                || unsafe {
                    runtime.blaslt().linear_bf16(
                        input.storage(),
                        weight.storage(),
                        reference.storage_mut(),
                        m,
                        n,
                        k,
                    )
                },
                || tiny_bf16_nt_into(&runtime, &input, &weight, &mut candidate),
            )?;

            println!(
                "tiny_bf16 site={} M={} N={} K={} cublaslt_mean_us={:.3} tiny_mean_us={:.3} mean_speedup={:.4}x cublaslt_p95_us={:.3} tiny_p95_us={:.3} p95_speedup={:.4}x rel_l2={:.8} cosine={:.8} max_abs={:.6}",
                site,
                m,
                n,
                k,
                stats.reference.mean_us,
                stats.candidate.mean_us,
                stats.speedup_mean,
                stats.reference.p95_us,
                stats.candidate.p95_us,
                stats.speedup_p95,
                rel_l2,
                cosine,
                max_abs,
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "GPU microbenchmark with rotating weights to exceed L2"]
fn bench_tiny_bf16_decode_shapes_rotating_weights() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let bench = BenchConfig {
        warmup: 12,
        batches: 30,
        // Keep this a multiple of ROTATING_WEIGHT_COUNT so every measured
        // batch sees the same number of launches for each weight tensor.
        iterations_per_batch: 8,
    };

    for (site, n, k) in [
        ("down", 2_048usize, 8_192usize),
        ("gate_up", 16_384usize, 2_048usize),
    ] {
        let mut weights = Vec::with_capacity(ROTATING_WEIGHT_COUNT);
        for weight_index in 0..ROTATING_WEIGHT_COUNT {
            weights.push(runtime.upload(
                &deterministic_values(n * k, 29 + weight_index * 2, 257),
                Shape::new([n, k]),
            )?);
        }
        let weight_bytes = n
            .checked_mul(k)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<bf16>()))
            .ok_or_else(|| anyhow::anyhow!("tiny BF16 rotating weight byte size overflow"))?;

        for m in [1usize, 2, 4, 8] {
            let input = runtime.upload(
                &deterministic_values(m * k, 17 + m, 251),
                Shape::new([m, k]),
            )?;
            let mut reference = runtime.alloc_uninit::<bf16>(Shape::new([m, n]))?;
            let mut candidate = runtime.alloc_uninit::<bf16>(Shape::new([m, n]))?;

            unsafe {
                runtime.blaslt().linear_bf16(
                    input.storage(),
                    weights[0].storage(),
                    reference.storage_mut(),
                    m,
                    n,
                    k,
                )?;
            }
            tiny_bf16_nt_into(&runtime, &input, &weights[0], &mut candidate)?;
            runtime.synchronize()?;
            let reference_host = runtime.download(&reference)?;
            let candidate_host = runtime.download(&candidate)?;
            let (rel_l2, cosine, max_abs) = output_metrics(&reference_host, &candidate_host);

            let mut reference_weight = 0usize;
            let mut candidate_weight = 0usize;
            let stats = benchmark_gpu_paired(
                runtime.context(),
                runtime.stream(),
                bench,
                || {
                    let weight = &weights[reference_weight];
                    reference_weight = (reference_weight + 1) % weights.len();
                    unsafe {
                        runtime.blaslt().linear_bf16(
                            input.storage(),
                            weight.storage(),
                            reference.storage_mut(),
                            m,
                            n,
                            k,
                        )
                    }
                },
                || {
                    let weight = &weights[candidate_weight];
                    candidate_weight = (candidate_weight + 1) % weights.len();
                    tiny_bf16_nt_into(&runtime, &input, weight, &mut candidate)
                },
            )?;

            println!(
                "tiny_bf16_rotating weights={} site={} M={} N={} K={} weight_mib={:.3} cublaslt_mean_us={:.3} tiny_mean_us={:.3} mean_speedup={:.4}x cublaslt_p95_us={:.3} tiny_p95_us={:.3} p95_speedup={:.4}x cublaslt_effective_gbps={:.3} tiny_effective_gbps={:.3} rel_l2={:.8} cosine={:.8} max_abs={:.6}",
                weights.len(),
                site,
                m,
                n,
                k,
                weight_bytes as f64 / (1024.0 * 1024.0),
                stats.reference.mean_us,
                stats.candidate.mean_us,
                stats.speedup_mean,
                stats.reference.p95_us,
                stats.candidate.p95_us,
                stats.speedup_p95,
                stats.reference.effective_gbps(weight_bytes),
                stats.candidate.effective_gbps(weight_bytes),
                rel_l2,
                cosine,
                max_abs,
            );
        }
    }
    Ok(())
}
