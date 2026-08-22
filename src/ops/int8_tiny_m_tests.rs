use anyhow::Result;
use half::bf16;

use crate::{
    cuda::{
        CudaRuntime,
        benchmark::{BenchConfig, benchmark_gpu_paired},
        testing::{assert_eq_bf16, readback},
    },
    tensor::Shape,
};

use super::{
    int8_tiny_m::{
        Int8TinyMWorkspace, linear_int8_tiny_m_into, linear_int8_tiny_m_prequantized_into,
        quantize_int8_tiny_m_input_into, quantize_weight_s8_per_channel,
        silu_mul_packed_bf16_to_int8_tiny_m_into,
    },
    linear::linear_bf16_into,
    silu_mul::silu_mul_packed_bf16_into,
};

fn patterned_bf16(elements: usize, mul: usize, modulus: usize, center: f32, scale: f32) -> Vec<bf16> {
    (0..elements)
        .map(|index| bf16::from_f32(((index * mul % modulus) as f32 - center) / scale))
        .collect()
}

fn cosine_similarity(actual: &[bf16], reference: &[bf16]) -> f64 {
    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut reference_norm = 0.0f64;
    for (&actual, &reference) in actual.iter().zip(reference) {
        let actual = f64::from(actual.to_f32());
        let reference = f64::from(reference.to_f32());
        dot += actual * reference;
        actual_norm += actual * actual;
        reference_norm += reference * reference;
    }
    dot / (actual_norm.sqrt() * reference_norm.sqrt())
}

fn relative_l2(actual: &[bf16], reference: &[bf16]) -> f64 {
    let mut error = 0.0f64;
    let mut reference_norm = 0.0f64;
    for (&actual, &reference) in actual.iter().zip(reference) {
        let actual = f64::from(actual.to_f32());
        let reference = f64::from(reference.to_f32());
        let delta = actual - reference;
        error += delta * delta;
        reference_norm += reference * reference;
    }
    (error / reference_norm).sqrt()
}

#[test]
fn int8_tiny_m_dp4a_matches_cpu_quantized_reference() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    const K: usize = 64;
    const N: usize = 32;
    let weight_host = patterned_bf16(N * K, 13, 89, 44.0, 32.0);
    let weight = runtime.upload(&weight_host, Shape::new([N, K]))?;
    let weight_q = quantize_weight_s8_per_channel(&runtime, &weight)?;
    let weight_i8 = readback(&runtime, &weight_q.data)?;
    let weight_scales = readback(&runtime, &weight_q.scales)?;

    for m in [1usize, 4, 8] {
        let input_host = patterned_bf16(m * K, 17 + m, 101, 50.0, 24.0);
        let input = runtime.upload(&input_host, Shape::new([m, K]))?;
        let mut workspace = Int8TinyMWorkspace::new(&runtime, 8, K)?;
        quantize_int8_tiny_m_input_into(&runtime, &input, &mut workspace)?;

        let mut output = runtime.alloc_bf16(Shape::new([m, N]))?;
        linear_int8_tiny_m_prequantized_into(&runtime, m, &weight_q, &workspace, &mut output)?;

        let mut input_i8 = runtime.alloc_uninit::<i8>(Shape::new([m, K]))?;
        let mut input_scales = runtime.alloc_uninit::<f32>(Shape::new([m]))?;
        unsafe {
            runtime.kernels().int8_tiny_m().launch_quantize_rows(
                runtime.stream(),
                crate::cuda::QuantizeS8RowsLaunch {
                    input: input.storage(),
                    output: input_i8.storage_mut(),
                    scales: input_scales.storage_mut(),
                    rows: m,
                    cols: K,
                },
            )?;
        }
        let input_i8 = readback(&runtime, &input_i8)?;
        let input_scales = readback(&runtime, &input_scales)?;
        let mut expected = Vec::with_capacity(m * N);
        for row in 0..m {
            for col in 0..N {
                let mut sum = 0i32;
                for k in 0..K {
                    sum += i32::from(input_i8[row * K + k]) * i32::from(weight_i8[col * K + k]);
                }
                let scale = input_scales[row] * weight_scales[col];
                expected.push(bf16::from_f32(sum as f32 * scale));
            }
        }
        assert_eq_bf16(&readback(&runtime, &output)?, &expected);
    }
    Ok(())
}

#[test]
fn int8_tiny_m_w8a8_quality_tracks_bf16() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    const K: usize = 256;
    const N: usize = 128;
    let weight = runtime.upload(
        &patterned_bf16(N * K, 13, 257, 128.0, 128.0),
        Shape::new([N, K]),
    )?;
    let weight_q = quantize_weight_s8_per_channel(&runtime, &weight)?;

    for m in [1usize, 2, 4, 8] {
        let input = runtime.upload(
            &patterned_bf16(m * K, 17 + m, 251, 125.0, 96.0),
            Shape::new([m, K]),
        )?;
        let mut reference = runtime.alloc_bf16(Shape::new([m, N]))?;
        linear_bf16_into(&runtime, &input, &weight, &mut reference)?;
        let mut workspace = Int8TinyMWorkspace::new(&runtime, 8, K)?;
        let mut candidate = runtime.alloc_bf16(Shape::new([m, N]))?;
        linear_int8_tiny_m_into(&runtime, &input, &weight_q, &mut workspace, &mut candidate)?;
        runtime.synchronize()?;

        let reference = readback(&runtime, &reference)?;
        let candidate = readback(&runtime, &candidate)?;
        let cosine = cosine_similarity(&candidate, &reference);
        let rel_l2 = relative_l2(&candidate, &reference);
        assert!(cosine >= 0.998, "M={m} cosine={cosine}");
        assert!(rel_l2 <= 0.07, "M={m} relative_l2={rel_l2}");
    }
    Ok(())
}

#[test]
fn fused_swiglu_dynamic_int8_matches_unfused_tail_bitwise() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    const K: usize = 256;
    const N: usize = 128;
    let weight = runtime.upload(
        &patterned_bf16(N * K, 13, 257, 128.0, 128.0),
        Shape::new([N, K]),
    )?;
    let weight_q = quantize_weight_s8_per_channel(&runtime, &weight)?;

    for m in [1usize, 2] {
        let packed = runtime.upload(
            &patterned_bf16(m * K * 2, 19 + m, 509, 254.0, 128.0),
            Shape::new([m, K * 2]),
        )?;
        let mut activated = runtime.alloc_bf16(Shape::new([m, K]))?;
        let mut unfused_workspace = Int8TinyMWorkspace::new(&runtime, 2, K)?;
        let mut fused_workspace = Int8TinyMWorkspace::new(&runtime, 2, K)?;
        let mut reference = runtime.alloc_bf16(Shape::new([m, N]))?;
        let mut candidate = runtime.alloc_bf16(Shape::new([m, N]))?;

        silu_mul_packed_bf16_into(&runtime, &packed, &mut activated)?;
        linear_int8_tiny_m_into(
            &runtime,
            &activated,
            &weight_q,
            &mut unfused_workspace,
            &mut reference,
        )?;
        silu_mul_packed_bf16_to_int8_tiny_m_into(&runtime, &packed, &mut fused_workspace)?;
        linear_int8_tiny_m_prequantized_into(
            &runtime,
            m,
            &weight_q,
            &fused_workspace,
            &mut candidate,
        )?;
        runtime.synchronize()?;
        assert_eq_bf16(&readback(&runtime, &candidate)?, &readback(&runtime, &reference)?);
    }
    Ok(())
}

#[test]
#[ignore = "GPU benchmark"]
fn bench_int8_tiny_m_down_proj_vs_bf16() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    const K: usize = 8192;
    const N: usize = 2048;
    let weight = runtime.upload(
        &patterned_bf16(N * K, 13, 521, 260.0, 512.0),
        Shape::new([N, K]),
    )?;
    let weight_q = quantize_weight_s8_per_channel(&runtime, &weight)?;
    runtime.synchronize()?;
    let config = BenchConfig {
        warmup: 8,
        batches: 24,
        iterations_per_batch: 10,
    };

    for m in [1usize, 2, 4, 8] {
        let input = runtime.upload(
            &patterned_bf16(m * K, 17 + m, 509, 254.0, 384.0),
            Shape::new([m, K]),
        )?;
        let mut bf16_output = runtime.alloc_bf16(Shape::new([m, N]))?;
        let mut int8_output = runtime.alloc_bf16(Shape::new([m, N]))?;
        let mut workspace = Int8TinyMWorkspace::new(&runtime, 8, K)?;

        linear_bf16_into(&runtime, &input, &weight, &mut bf16_output)?;
        linear_int8_tiny_m_into(&runtime, &input, &weight_q, &mut workspace, &mut int8_output)?;
        runtime.synchronize()?;
        let reference = readback(&runtime, &bf16_output)?;
        let candidate = readback(&runtime, &int8_output)?;
        let cosine = cosine_similarity(&candidate, &reference);
        let rel_l2 = relative_l2(&candidate, &reference);

        quantize_int8_tiny_m_input_into(&runtime, &input, &mut workspace)?;
        runtime.synchronize()?;
        let gemm_only = benchmark_gpu_paired(
            runtime.context(),
            runtime.stream(),
            config,
            || linear_bf16_into(&runtime, &input, &weight, &mut bf16_output),
            || {
                linear_int8_tiny_m_prequantized_into(
                    &runtime,
                    m,
                    &weight_q,
                    &workspace,
                    &mut int8_output,
                )
            },
        )?;

        let end_to_end = benchmark_gpu_paired(
            runtime.context(),
            runtime.stream(),
            config,
            || linear_bf16_into(&runtime, &input, &weight, &mut bf16_output),
            || linear_int8_tiny_m_into(&runtime, &input, &weight_q, &mut workspace, &mut int8_output),
        )?;

        println!(
            "int8_tiny_m_down m={} cosine={:.6} rel_l2={:.6} gemm_bf16_mean_us={:.3} gemm_int8_mean_us={:.3} gemm_speedup={:.4}x e2e_bf16_mean_us={:.3} e2e_int8_mean_us={:.3} e2e_speedup={:.4}x e2e_int8_p95_us={:.3}",
            m,
            cosine,
            rel_l2,
            gemm_only.reference.mean_us,
            gemm_only.candidate.mean_us,
            gemm_only.speedup_mean,
            end_to_end.reference.mean_us,
            end_to_end.candidate.mean_us,
            end_to_end.speedup_mean,
            end_to_end.candidate.p95_us,
        );
    }
    Ok(())
}

#[test]
#[ignore = "GPU benchmark"]
fn bench_fused_swiglu_dynamic_int8_tail() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    const K: usize = 8192;
    const N: usize = 2048;
    let weight = runtime.upload(
        &patterned_bf16(N * K, 13, 521, 260.0, 512.0),
        Shape::new([N, K]),
    )?;
    let weight_q = quantize_weight_s8_per_channel(&runtime, &weight)?;
    let config = BenchConfig {
        warmup: 8,
        batches: 24,
        iterations_per_batch: 10,
    };

    for m in [1usize, 2] {
        let packed = runtime.upload(
            &patterned_bf16(m * K * 2, 19 + m, 1021, 510.0, 256.0),
            Shape::new([m, K * 2]),
        )?;
        let mut activated = runtime.alloc_bf16(Shape::new([m, K]))?;
        let mut unfused_workspace = Int8TinyMWorkspace::new(&runtime, 2, K)?;
        let mut fused_workspace = Int8TinyMWorkspace::new(&runtime, 2, K)?;
        let mut unfused_output = runtime.alloc_bf16(Shape::new([m, N]))?;
        let mut fused_output = runtime.alloc_bf16(Shape::new([m, N]))?;

        let quant_stage = benchmark_gpu_paired(
            runtime.context(),
            runtime.stream(),
            config,
            || {
                silu_mul_packed_bf16_into(&runtime, &packed, &mut activated)?;
                quantize_int8_tiny_m_input_into(&runtime, &activated, &mut unfused_workspace)
            },
            || silu_mul_packed_bf16_to_int8_tiny_m_into(&runtime, &packed, &mut fused_workspace),
        )?;

        let full_tail = benchmark_gpu_paired(
            runtime.context(),
            runtime.stream(),
            config,
            || {
                silu_mul_packed_bf16_into(&runtime, &packed, &mut activated)?;
                linear_int8_tiny_m_into(
                    &runtime,
                    &activated,
                    &weight_q,
                    &mut unfused_workspace,
                    &mut unfused_output,
                )
            },
            || {
                silu_mul_packed_bf16_to_int8_tiny_m_into(
                    &runtime,
                    &packed,
                    &mut fused_workspace,
                )?;
                linear_int8_tiny_m_prequantized_into(
                    &runtime,
                    m,
                    &weight_q,
                    &fused_workspace,
                    &mut fused_output,
                )
            },
        )?;

        println!(
            "int8_fused_swiglu m={} unfused_quant_mean_us={:.3} fused_quant_mean_us={:.3} quant_speedup={:.4}x unfused_tail_mean_us={:.3} fused_tail_mean_us={:.3} tail_speedup={:.4}x fused_tail_p95_us={:.3}",
            m,
            quant_stage.reference.mean_us,
            quant_stage.candidate.mean_us,
            quant_stage.speedup_mean,
            full_tail.reference.mean_us,
            full_tail.candidate.mean_us,
            full_tail.speedup_mean,
            full_tail.candidate.p95_us,
        );
    }
    Ok(())
}
