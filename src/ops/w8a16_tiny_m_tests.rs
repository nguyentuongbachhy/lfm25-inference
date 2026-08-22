use anyhow::Result;
use half::bf16;

use crate::{cuda::CudaRuntime, tensor::Shape};

use super::{
    linear::linear_bf16_into,
    int8_tiny_m::{linear_w8a16_tiny_m_into, quantize_weight_s8_per_channel},
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
    dot / (actual_norm * reference_norm)
        .sqrt()
        .max(f64::MIN_POSITIVE)
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
    (error / reference_norm.max(f64::MIN_POSITIVE)).sqrt()
}

#[test]
fn w8a16_tiny_m_matches_quantized_weight_reference() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    const K: usize = 64;
    const N: usize = 32;
    let weight_host = patterned_bf16(N * K, 13, 89, 44.0, 32.0);
    let weight = runtime.upload(&weight_host, Shape::new([N, K]))?;
    let weight_q = quantize_weight_s8_per_channel(&runtime, &weight)?;
    let weight_i8 = runtime.download(&weight_q.data)?;
    let weight_scales = runtime.download(&weight_q.scales)?;

    for m in [1usize, 2] {
        let input_host = patterned_bf16(m * K, 17 + m, 101, 50.0, 24.0);
        let input = runtime.upload(&input_host, Shape::new([m, K]))?;
        let mut output = runtime.alloc_bf16(Shape::new([m, N]))?;
        linear_w8a16_tiny_m_into(&runtime, &input, &weight_q, &mut output)?;
        runtime.synchronize()?;
        let actual = runtime.download(&output)?;

        let mut reference = Vec::with_capacity(m * N);
        for row in 0..m {
            for col in 0..N {
                let mut sum = 0.0f32;
                for k in 0..K {
                    sum += input_host[row * K + k].to_f32()
                        * f32::from(weight_i8[col * K + k]);
                }
                reference.push(bf16::from_f32(sum * weight_scales[col]));
            }
        }

        let cosine = cosine_similarity(&actual, &reference);
        let rel_l2 = relative_l2(&actual, &reference);
        assert!(cosine >= 0.99999, "M={m} cosine={cosine}");
        assert!(rel_l2 <= 0.005, "M={m} relative_l2={rel_l2}");
    }
    Ok(())
}

#[test]
fn w8a16_tiny_m_quality_tracks_bf16_better_than_w8a8_smoke_gate() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    const K: usize = 256;
    const N: usize = 128;
    let weight = runtime.upload(
        &patterned_bf16(N * K, 13, 257, 128.0, 128.0),
        Shape::new([N, K]),
    )?;
    let weight_q = quantize_weight_s8_per_channel(&runtime, &weight)?;

    for m in [1usize, 2] {
        let input = runtime.upload(
            &patterned_bf16(m * K, 17 + m, 251, 125.0, 96.0),
            Shape::new([m, K]),
        )?;
        let mut reference = runtime.alloc_bf16(Shape::new([m, N]))?;
        let mut candidate = runtime.alloc_bf16(Shape::new([m, N]))?;
        linear_bf16_into(&runtime, &input, &weight, &mut reference)?;
        linear_w8a16_tiny_m_into(&runtime, &input, &weight_q, &mut candidate)?;
        runtime.synchronize()?;

        let reference = runtime.download(&reference)?;
        let candidate = runtime.download(&candidate)?;
        let cosine = cosine_similarity(&candidate, &reference);
        let rel_l2 = relative_l2(&candidate, &reference);
        assert!(cosine >= 0.9995, "M={m} cosine={cosine}");
        assert!(rel_l2 <= 0.035, "M={m} relative_l2={rel_l2}");
    }
    Ok(())
}
