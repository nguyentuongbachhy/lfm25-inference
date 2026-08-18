use anyhow::Result;
use half::bf16;

use crate::{
    cuda::{
        CudaRuntime,
        benchmark::{BenchConfig, benchmark_gpu},
        testing::{assert_close_bf16, readback},
    },
    tensor::Shape,
};

fn linear_reference(x: &[bf16], weight: &[bf16], m: usize, n: usize, k: usize) -> Vec<bf16> {
    let mut out = vec![bf16::from_f32(0.0); m * n];
    for row in 0..m {
        for out_feature in 0..n {
            let mut acc = 0.0f32;
            for inner in 0..k {
                acc += x[row * k + inner].to_f32() * weight[out_feature * k + inner].to_f32();
            }
            out[row * n + out_feature] = bf16::from_f32(acc);
        }
    }
    out
}

#[test]
fn linear_bf16_layout_correctness() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    const M: usize = 2;
    const K: usize = 3;
    const N: usize = 4;
    let x_host = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(bf16::from_f32);
    let weight_host = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 1.0, 1.0,
    ]
    .map(bf16::from_f32);
    let expected = linear_reference(&x_host, &weight_host, M, N, K);
    let x = runtime.upload(&x_host, Shape::new([M, K]))?;
    let weight = runtime.upload(&weight_host, Shape::new([N, K]))?;
    let mut out = runtime.zeros::<bf16>(Shape::new([M, N]))?;
    unsafe {
        runtime
            .blaslt()
            .linear_bf16(x.storage(), weight.storage(), out.storage_mut(), M, N, K)?;
    }
    let actual = readback(&runtime, &out)?;
    assert_close_bf16(&actual, &expected, 0.01, 0.01);
    Ok(())
}

#[test]
fn linear_bf16_various_shapes() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    for (m, n, k) in [
        (1, 1, 1),
        (1, 7, 3),
        (3, 5, 7),
        (7, 9, 16),
        (16, 13, 31),
        (32, 64, 128),
    ] {
        let x_host: Vec<bf16> = (0usize..m * k)
            .map(|i| {
                let value = ((i.wrapping_mul(17) % 31) as f32 - 15.0) / 16.0;
                bf16::from_f32(value)
            })
            .collect();
        let weight_host: Vec<bf16> = (0..n * k)
            .map(|i| {
                let value = ((i.wrapping_mul(13) % 29) as f32 - 14.0) / 16.0;
                bf16::from_f32(value)
            })
            .collect();
        let expected = linear_reference(&x_host, &weight_host, m, n, k);
        let x = runtime.upload(&x_host, Shape::new([m, k]))?;
        let weight = runtime.upload(&weight_host, Shape::new([n, k]))?;
        let mut out = runtime.zeros::<bf16>(Shape::new([m, n]))?;
        unsafe {
            runtime.blaslt().linear_bf16(
                x.storage(),
                weight.storage(),
                out.storage_mut(),
                m,
                n,
                k,
            )?;
        }
        let actual = readback(&runtime, &out)?;
        assert_close_bf16(&actual, &expected, 0.02, 0.02);
    }
    Ok(())
}

#[test]
fn linear_bf16_reuses_cached_plan() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    const M: usize = 4;
    const N: usize = 8;
    const K: usize = 16;
    let x = runtime.upload(&[bf16::from_f32(1.0); M * K], Shape::new([M, K]))?;
    let weight = runtime.upload(&[bf16::from_f32(0.5); N * K], Shape::new([N, K]))?;
    let mut out = runtime.zeros::<bf16>(Shape::new([M, N]))?;
    assert_eq!(runtime.blaslt().cached_plan_count(), 0);
    unsafe {
        runtime
            .blaslt()
            .linear_bf16(x.storage(), weight.storage(), out.storage_mut(), M, N, K)?;
    }
    assert_eq!(runtime.blaslt().cached_plan_count(), 1);
    unsafe {
        runtime
            .blaslt()
            .linear_bf16(x.storage(), weight.storage(), out.storage_mut(), M, N, K)?;
    }
    assert_eq!(runtime.blaslt().cached_plan_count(), 1);
    Ok(())
}

#[test]
fn linear_bf16_caches_per_shape() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    runtime.blaslt().prepare_linear_bf16(1, 2048, 2048)?;
    assert_eq!(runtime.blaslt().cached_plan_count(), 1);
    runtime.blaslt().prepare_linear_bf16(16, 2048, 2048)?;
    assert_eq!(runtime.blaslt().cached_plan_count(), 2);
    runtime.blaslt().prepare_linear_bf16(1, 2048, 2048)?;
    assert_eq!(runtime.blaslt().cached_plan_count(), 2);
    Ok(())
}

#[test]
#[ignore = "GPU benchmark"]
fn bench_linear_bf16() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    const K: usize = 2048;
    const N: usize = 2048;
    let weight_host: Vec<bf16> = (0..N * K)
        .map(|i| {
            let value = ((i % 127) as f32 - 63.0) / 128.0;
            bf16::from_f32(value)
        })
        .collect();
    let weight = runtime.upload(&weight_host, Shape::new([N, K]))?;
    let bench_config = BenchConfig {
        warmup: 20,
        batches: 50,
        iterations_per_batch: 10,
    };
    for m in [1, 4, 16, 64, 256, 1024] {
        let x_host: Vec<bf16> = (0..m * K)
            .map(|i| {
                let value = ((i % 61) as f32 - 30.0) / 64.0;
                bf16::from_f32(value)
            })
            .collect();
        let x = runtime.upload(&x_host, Shape::new([m, K]))?;
        let mut out = runtime.zeros::<bf16>(Shape::new([m, N]))?;
        let stats = benchmark_gpu(runtime.context(), runtime.stream(), bench_config, || {
            unsafe {
                runtime.blaslt().linear_bf16(
                    x.storage(),
                    weight.storage(),
                    out.storage_mut(),
                    m,
                    N,
                    K,
                )?;
            }
            Ok(())
        })?;
        let seconds = stats.mean_us * 1e-6;
        let flops = 2.0 * m as f64 * N as f64 * K as f64;
        let tflops = flops / seconds / 1e12;
        println!(
            "M={m:>4} K={K} N={N} | mean={:>9.3} us | p50={:>9.3} us | p95={:>9.3} us | min={:>9.3} us | {:>8.2} TFLOP/s",
            stats.mean_us, stats.p50_us, stats.p95_us, stats.min_us, tflops,
        );
    }
    Ok(())
}
