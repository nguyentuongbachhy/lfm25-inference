use anyhow::Result;
use half::bf16;

use crate::{
    cuda::{
        CudaRuntime,
        benchmark::{BenchConfig, BenchStats, benchmark_gpu},
        blaslt::{Fp8LinearConfig, fp8::Fp8ScaleMode},
        testing::readback,
    },
    tensor::Shape,
};

fn fp8_config(m: usize, n: usize, k: usize, scale_mode: Fp8ScaleMode) -> Fp8LinearConfig {
    Fp8LinearConfig {
        m,
        n,
        k,
        scale_mode,
        output_scale: 1.0,
    }
}

#[test]
fn linear_fp8_e4m3_correctness() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    const M: usize = 1;
    const N: usize = 16;
    const K: usize = 32;

    let x_host: Vec<u8> = (0..M * K)
        .map(|index| if index % 2 == 0 { 0x38 } else { 0xb8 })
        .collect();
    let weight_host: Vec<u8> = (0..N * K)
        .map(|index| if index % 2 == 0 { 0x38 } else { 0xb8 })
        .collect();
    let x = runtime.upload(&x_host, Shape::new([M, K]))?;
    let weight = runtime.upload(&weight_host, Shape::new([N, K]))?;

    for scale_mode in [Fp8ScaleMode::Tensorwide, Fp8ScaleMode::Block32] {
        let mut out = runtime.zeros::<bf16>(Shape::new([M, N]))?;
        unsafe {
            runtime.blaslt().linear_fp8(
                x.storage(),
                weight.storage(),
                out.storage_mut(),
                fp8_config(M, N, K, scale_mode),
            )?;
        }
        let actual = readback(&runtime, &out)?;
        for value in actual {
            assert_eq!(value.to_f32(), K as f32);
        }
    }
    assert_eq!(runtime.blaslt().cached_fp8_plan_count(), 2);
    Ok(())
}

#[test]
fn quantize_bf16_to_e4m3_correctness() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let input_host = [
        bf16::from_f32(1.0),
        bf16::from_f32(-1.0),
        bf16::from_f32(0.5),
        bf16::from_f32(-0.5),
    ];
    let input = runtime.upload(&input_host, Shape::new([1, 4]))?;
    let mut output = runtime.zeros::<u8>(Shape::new([1, 4]))?;
    unsafe {
        runtime.kernels().fp8_quantize().launch_bf16_e4m3(
            runtime.stream(),
            input.storage(),
            output.storage_mut(),
            4,
            1.0,
        )?;
    }
    let actual = readback(&runtime, &output)?;
    assert_eq!(actual, [0x38, 0xb8, 0x30, 0xb0]);
    Ok(())
}

#[test]
fn tensorwide_fp8_numerical_error_is_bounded() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    const M: usize = 4;
    const N: usize = 64;
    const K: usize = 128;

    fn deterministic_value(index: usize, seed: u32) -> bf16 {
        let mut value = (index as u32).wrapping_add(seed);
        value ^= value >> 16;
        value = value.wrapping_mul(0x7feb_352d);
        value ^= value >> 15;
        value = value.wrapping_mul(0x846c_a68b);
        value ^= value >> 16;
        let unit = (value & 0xffff) as f32 / 65_535.0;
        bf16::from_f32((unit - 0.5) * 1.5)
    }

    let x_host: Vec<bf16> = (0..M * K)
        .map(|index| deterministic_value(index, 0x1234_5678))
        .collect();
    let weight_host: Vec<bf16> = (0..N * K)
        .map(|index| deterministic_value(index, 0x9abc_def0))
        .collect();
    let x = runtime.upload(&x_host, Shape::new([M, K]))?;
    let weight = runtime.upload(&weight_host, Shape::new([N, K]))?;
    let mut x_fp8 = runtime.zeros::<u8>(Shape::new([M, K]))?;
    let mut weight_fp8 = runtime.zeros::<u8>(Shape::new([N, K]))?;
    let mut bf16_out = runtime.zeros::<bf16>(Shape::new([M, N]))?;
    let mut fp8_out = runtime.zeros::<bf16>(Shape::new([M, N]))?;

    unsafe {
        runtime.kernels().fp8_quantize().launch_bf16_e4m3(
            runtime.stream(),
            x.storage(),
            x_fp8.storage_mut(),
            M * K,
            1.0,
        )?;
        runtime.kernels().fp8_quantize().launch_bf16_e4m3(
            runtime.stream(),
            weight.storage(),
            weight_fp8.storage_mut(),
            N * K,
            1.0,
        )?;
        runtime.blaslt().linear_bf16(
            x.storage(),
            weight.storage(),
            bf16_out.storage_mut(),
            M,
            N,
            K,
        )?;
        runtime.blaslt().linear_fp8(
            x_fp8.storage(),
            weight_fp8.storage(),
            fp8_out.storage_mut(),
            fp8_config(M, N, K, Fp8ScaleMode::Tensorwide),
        )?;
    }

    let reference = readback(&runtime, &bf16_out)?;
    let actual = readback(&runtime, &fp8_out)?;
    let mut squared_error = 0.0f64;
    let mut squared_reference = 0.0f64;
    let mut dot = 0.0f64;
    let mut squared_actual = 0.0f64;
    for (expected, observed) in reference.iter().zip(&actual) {
        let expected = f64::from(expected.to_f32());
        let observed = f64::from(observed.to_f32());
        let difference = observed - expected;
        squared_error += difference * difference;
        squared_reference += expected * expected;
        squared_actual += observed * observed;
        dot += expected * observed;
    }
    let nrmse = (squared_error / squared_reference).sqrt();
    let cosine = dot / (squared_reference * squared_actual).sqrt();
    println!("tensorwide_fp8_nrmse={nrmse:.6}, cosine={cosine:.6}");
    assert!(nrmse < 0.08, "FP8 NRMSE {nrmse} exceeds 0.08");
    assert!(cosine > 0.995, "FP8 cosine {cosine} is below 0.995");
    Ok(())
}

#[derive(Clone, Copy)]
struct GemmShape {
    name: &'static str,
    n: usize,
    k: usize,
}

const LFM_GEMMS: [GemmShape; 5] = [
    GemmShape {
        name: "mlp_gate_up",
        n: 16_384,
        k: 2_048,
    },
    GemmShape {
        name: "mlp_down",
        n: 2_048,
        k: 8_192,
    },
    GemmShape {
        name: "lm_head",
        n: 65_536,
        k: 2_048,
    },
    GemmShape {
        name: "attention_qkv",
        n: 3_072,
        k: 2_048,
    },
    GemmShape {
        name: "attention_o",
        n: 2_048,
        k: 2_048,
    },
];

const M_SWEEP: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
const WEIGHT_RING_TARGET_BYTES: usize = 128 * 1024 * 1024;

fn weight_ring_count(elements: usize, element_size: usize) -> Result<usize> {
    let bytes = elements
        .checked_mul(element_size)
        .ok_or_else(|| anyhow::anyhow!("weight byte count overflow"))?;
    Ok(WEIGHT_RING_TARGET_BYTES.div_ceil(bytes).max(1))
}

fn e4m3_value(index: usize) -> (u8, bf16) {
    match index % 4 {
        0 => (0x30, bf16::from_f32(0.5)),
        1 => (0xb0, bf16::from_f32(-0.5)),
        2 => (0x28, bf16::from_f32(0.25)),
        _ => (0xa8, bf16::from_f32(-0.25)),
    }
}

fn report(
    shape: GemmShape,
    m: usize,
    precision: &str,
    stats: &BenchStats,
    bytes: usize,
) -> Result<()> {
    let flops = m
        .checked_mul(shape.n)
        .and_then(|value| value.checked_mul(shape.k))
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| anyhow::anyhow!("GEMM FLOP count overflow"))?;
    println!(
        "{},{},{},{:.3},{:.3},{:.3},{:.3},{:.2},{:.3}",
        shape.name,
        m,
        precision,
        stats.mean_us,
        stats.p50_us,
        stats.p95_us,
        stats.min_us,
        stats.effective_gbps(bytes),
        stats.effective_tflops(flops),
    );
    Ok(())
}

fn report_quantization(
    shape: GemmShape,
    m: usize,
    precision: &str,
    stats: &BenchStats,
    elements: usize,
) -> Result<()> {
    let bytes = elements
        .checked_mul(3)
        .ok_or_else(|| anyhow::anyhow!("quantization byte count overflow"))?;
    println!(
        "{},{},{},{:.3},{:.3},{:.3},{:.3},{:.2},0.000",
        shape.name,
        m,
        precision,
        stats.mean_us,
        stats.p50_us,
        stats.p95_us,
        stats.min_us,
        stats.effective_gbps(bytes),
    );
    Ok(())
}

#[test]
#[ignore = "GPU benchmark: exhaustive LFM BF16/FP8/MXFP8 matrix"]
fn bench_lfm_narrow_precision_gemms() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let config = BenchConfig {
        warmup: 10,
        batches: 20,
        iterations_per_batch: 5,
    };

    println!("shape,m,precision,mean_us,p50_us,p95_us,min_us,effective_gbps,tflops");

    for shape in LFM_GEMMS {
        let element_count = shape
            .n
            .checked_mul(shape.k)
            .ok_or_else(|| anyhow::anyhow!("weight element count overflow"))?;
        let mut weight_bf16_host = Vec::with_capacity(element_count);
        for index in 0..element_count {
            let (_, bf16) = e4m3_value(index);
            weight_bf16_host.push(bf16);
        }

        let bf16_ring_count = weight_ring_count(element_count, 2)?;
        let fp8_ring_count = weight_ring_count(element_count, 1)?;
        let mut weight_bf16 = Vec::with_capacity(bf16_ring_count);
        let mut weight_fp8 = Vec::with_capacity(fp8_ring_count);
        for _ in 0..bf16_ring_count {
            weight_bf16.push(runtime.upload(&weight_bf16_host, Shape::new([shape.n, shape.k]))?);
        }
        for _ in 0..fp8_ring_count {
            weight_fp8.push(runtime.zeros::<u8>(Shape::new([shape.n, shape.k]))?);
        }
        drop(weight_bf16_host);

        println!(
            "# {},bf16_weight_ring={},fp8_weight_ring={},target_bytes={}",
            shape.name, bf16_ring_count, fp8_ring_count, WEIGHT_RING_TARGET_BYTES,
        );

        let weight_quantize_stats =
            benchmark_gpu(runtime.context(), runtime.stream(), config, || unsafe {
                runtime.kernels().fp8_quantize().launch_bf16_e4m3(
                    runtime.stream(),
                    weight_bf16[0].storage(),
                    weight_fp8[0].storage_mut(),
                    element_count,
                    1.0,
                )
            })?;
        report_quantization(
            shape,
            0,
            "weight_quantize_offline",
            &weight_quantize_stats,
            element_count,
        )?;

        for index in 1..fp8_ring_count {
            unsafe {
                runtime.kernels().fp8_quantize().launch_bf16_e4m3(
                    runtime.stream(),
                    weight_bf16[index % bf16_ring_count].storage(),
                    weight_fp8[index].storage_mut(),
                    element_count,
                    1.0,
                )?;
            }
        }
        runtime.stream().synchronize()?;

        for m in M_SWEEP {
            let input_elements = m
                .checked_mul(shape.k)
                .ok_or_else(|| anyhow::anyhow!("input element count overflow"))?;
            let mut x_fp8_host = Vec::with_capacity(input_elements);
            let mut x_bf16_host = Vec::with_capacity(input_elements);
            for index in 0..input_elements {
                let (fp8, bf16) = e4m3_value(index.wrapping_mul(3));
                x_fp8_host.push(fp8);
                x_bf16_host.push(bf16);
            }

            let x_fp8 = runtime.upload(&x_fp8_host, Shape::new([m, shape.k]))?;
            let x_bf16 = runtime.upload(&x_bf16_host, Shape::new([m, shape.k]))?;
            let mut x_quantized = runtime.zeros::<u8>(Shape::new([m, shape.k]))?;
            let mut out = runtime.zeros::<bf16>(Shape::new([m, shape.n]))?;

            runtime.blaslt().prepare_linear_bf16(m, shape.n, shape.k)?;
            runtime
                .blaslt()
                .prepare_linear_fp8(m, shape.n, shape.k, Fp8ScaleMode::Tensorwide)?;
            runtime
                .blaslt()
                .prepare_linear_fp8(m, shape.n, shape.k, Fp8ScaleMode::Block32)?;
            runtime.stream().synchronize()?;

            let mut bf16_weight_index = 0usize;
            let bf16_stats =
                benchmark_gpu(runtime.context(), runtime.stream(), config, || unsafe {
                    let weight_index = bf16_weight_index;
                    bf16_weight_index = (bf16_weight_index + 1) % bf16_ring_count;
                    runtime.blaslt().linear_bf16(
                        x_bf16.storage(),
                        weight_bf16[weight_index].storage(),
                        out.storage_mut(),
                        m,
                        shape.n,
                        shape.k,
                    )
                })?;
            let mut tensorwide_weight_index = 0usize;
            let tensorwide_stats =
                benchmark_gpu(runtime.context(), runtime.stream(), config, || unsafe {
                    let weight_index = tensorwide_weight_index;
                    tensorwide_weight_index = (tensorwide_weight_index + 1) % fp8_ring_count;
                    runtime.blaslt().linear_fp8(
                        x_fp8.storage(),
                        weight_fp8[weight_index].storage(),
                        out.storage_mut(),
                        fp8_config(m, shape.n, shape.k, Fp8ScaleMode::Tensorwide),
                    )
                })?;
            let mut mxfp8_weight_index = 0usize;
            let mxfp8_stats =
                benchmark_gpu(runtime.context(), runtime.stream(), config, || unsafe {
                    let weight_index = mxfp8_weight_index;
                    mxfp8_weight_index = (mxfp8_weight_index + 1) % fp8_ring_count;
                    runtime.blaslt().linear_fp8(
                        x_fp8.storage(),
                        weight_fp8[weight_index].storage(),
                        out.storage_mut(),
                        fp8_config(m, shape.n, shape.k, Fp8ScaleMode::Block32),
                    )
                })?;
            let quantize_stats =
                benchmark_gpu(runtime.context(), runtime.stream(), config, || unsafe {
                    runtime.kernels().fp8_quantize().launch_bf16_e4m3(
                        runtime.stream(),
                        x_bf16.storage(),
                        x_quantized.storage_mut(),
                        input_elements,
                        1.0,
                    )
                })?;
            let mut total_weight_index = 0usize;
            let quantize_gemm_stats =
                benchmark_gpu(runtime.context(), runtime.stream(), config, || unsafe {
                    runtime.kernels().fp8_quantize().launch_bf16_e4m3(
                        runtime.stream(),
                        x_bf16.storage(),
                        x_quantized.storage_mut(),
                        input_elements,
                        1.0,
                    )?;
                    let weight_index = total_weight_index;
                    total_weight_index = (total_weight_index + 1) % fp8_ring_count;
                    runtime.blaslt().linear_fp8(
                        x_quantized.storage(),
                        weight_fp8[weight_index].storage(),
                        out.storage_mut(),
                        fp8_config(m, shape.n, shape.k, Fp8ScaleMode::Tensorwide),
                    )
                })?;

            let bf16_bytes = (input_elements + element_count + m * shape.n) * 2;
            let fp8_bytes = input_elements + element_count + m * shape.n * 2;
            let mx_scale_bytes = input_elements / 32 + element_count / 32;

            report(shape, m, "bf16", &bf16_stats, bf16_bytes)?;
            report(shape, m, "fp8_e4m3", &tensorwide_stats, fp8_bytes + 8)?;
            report(
                shape,
                m,
                "mxfp8_block32",
                &mxfp8_stats,
                fp8_bytes + mx_scale_bytes,
            )?;
            report_quantization(
                shape,
                m,
                "activation_quantize",
                &quantize_stats,
                input_elements,
            )?;
            report(
                shape,
                m,
                "fp8_quantize_gemm",
                &quantize_gemm_stats,
                fp8_bytes + input_elements * 3 + 8,
            )?;

            if m == 1 && matches!(shape.name, "mlp_gate_up" | "mlp_down") {
                let speedup = bf16_stats.mean_us / quantize_gemm_stats.mean_us;
                println!(
                    "# gate,{},m=1,fp8_quantize_gemm_speedup={:.3}",
                    shape.name, speedup,
                );
                assert!(
                    speedup >= 1.3,
                    "{} M=1 FP8 quantize+GEMM speedup {:.3} is below 1.3",
                    shape.name,
                    speedup,
                );
            }
        }
    }

    Ok(())
}
