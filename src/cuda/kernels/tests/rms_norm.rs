use anyhow::Result;
use half::bf16;

use crate::{
    cuda::{
        CudaRuntime, RmsNormLaunch,
        benchmark::{BenchConfig, benchmark_gpu},
        testing::{assert_close_bf16, readback},
    },
    tensor::Shape,
};

const EPS: f32 = 1e-6;

fn make_input(rows: usize, hidden_size: usize) -> Vec<bf16> {
    (0..rows * hidden_size)
        .map(|i| {
            let value = ((i.wrapping_mul(37) % 1024) as f32 - 512.0) / 256.0;
            bf16::from_f32(value)
        })
        .collect()
}

fn make_weight(hidden_size: usize) -> Vec<bf16> {
    (0..hidden_size)
        .map(|i| {
            let value = 0.5 + (i % 257) as f32 / 256.0;
            bf16::from_f32(value)
        })
        .collect()
}

fn rms_norm_reference(
    x: &[bf16],
    weight: &[bf16],
    rows: usize,
    hidden_size: usize,
    eps: f32,
) -> Vec<bf16> {
    let mut out = Vec::with_capacity(x.len());

    for row in 0..rows {
        let start = row * hidden_size;
        let end = start + hidden_size;
        let row_x = &x[start..end];
        let mut sum_sq = 0.0f32;

        for &value in row_x {
            let value = value.to_f32();
            sum_sq += value * value;
        }

        let variance = sum_sq / hidden_size as f32;
        let inv_rms = 1.0 / (variance + eps).sqrt();

        for i in 0..hidden_size {
            // Match HF LFM2 ordering:
            //
            // FP32 normalize
            // -> BF16 cast
            // -> BF16 weight multiply
            let normalized = bf16::from_f32(row_x[i].to_f32() * inv_rms);
            let output = bf16::from_f32(normalized.to_f32() * weight[i].to_f32());
            out.push(output);
        }
    }

    out
}

fn run_rms_norm_case(runtime: &CudaRuntime, rows: usize, hidden_size: usize) -> Result<()> {
    let x_host = make_input(rows, hidden_size);
    let weight_host = make_weight(hidden_size);
    let expected = rms_norm_reference(&x_host, &weight_host, rows, hidden_size, EPS);
    let x = runtime.upload(&x_host, Shape::new([rows, hidden_size]))?;
    let weight = runtime.upload(&weight_host, Shape::new([hidden_size]))?;
    let mut out = runtime.zeros::<bf16>(Shape::new([rows, hidden_size]))?;

    unsafe {
        runtime.kernels().rms_norm().launch_bf16(
            runtime.stream(),
            RmsNormLaunch {
                x: x.storage(),
                weight: weight.storage(),
                out: out.storage_mut(),
                rows,
                hidden_size,
                eps: EPS,
            },
        )?;
    }

    let actual = readback(runtime, &out)?;
    assert_close_bf16(&actual, &expected, 0.01, 0.01);
    Ok(())
}

#[test]
fn rms_norm_bf16_correctness() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;

    for rows in [1, 3, 16] {
        for hidden_size in [
            1, 2, 3, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025,
            2047, 2048, 2049,
        ] {
            run_rms_norm_case(&runtime, rows, hidden_size)?;
        }
    }

    Ok(())
}

#[test]
#[ignore = "GPU benchmark"]
fn bench_rms_norm_bf16() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    const HIDDEN_SIZE: usize = 2048;
    let weight_host = make_weight(HIDDEN_SIZE);
    let weight = runtime.upload(&weight_host, Shape::new([HIDDEN_SIZE]))?;

    for rows in [1, 4, 16, 64, 256, 1024, 4096] {
        let x_host = make_input(rows, HIDDEN_SIZE);
        let x = runtime.upload(&x_host, Shape::new([rows, HIDDEN_SIZE]))?;
        let mut out = runtime.zeros::<bf16>(Shape::new([rows, HIDDEN_SIZE]))?;

        let stats = benchmark_gpu(
            runtime.context(),
            runtime.stream(),
            BenchConfig::default(),
            || {
                unsafe {
                    runtime.kernels().rms_norm().launch_bf16(
                        runtime.stream(),
                        RmsNormLaunch {
                            x: x.storage(),
                            weight: weight.storage(),
                            out: out.storage_mut(),
                            rows,
                            hidden_size: HIDDEN_SIZE,
                            eps: EPS,
                        },
                    )?;
                }
                Ok(())
            },
        )?;

        // Approximate logical traffic:
        // x read      = 2 bytes
        // weight read = 2 bytes
        // out write   = 2 bytes
        // per element = 6 bytes
        // x is read twice by this implementation, so physical traffic
        // is closer to 8 bytes/element before caching effects.
        let logical_bytes = rows as f64 * HIDDEN_SIZE as f64 * 6.0;
        let physical_bytes = rows as f64 * HIDDEN_SIZE as f64 * 8.0;
        let seconds = stats.mean_us * 1e-6;
        let logical_gbps = logical_bytes / seconds / 1e9;
        let physical_gbps = physical_bytes / seconds / 1e9;
        let rows_per_second = rows as f64 / seconds;

        println!(
            "rows={rows:>5} | mean={:>8.3} us | p50={:>8.3} us | p95={:>8.3} us | min={:>8.3} us | logical={:>8.2} GB/s | physical={:>8.2} GB/s | {:>12.0} rows/s",
            stats.mean_us,
            stats.p50_us,
            stats.p95_us,
            stats.min_us,
            logical_gbps,
            physical_gbps,
            rows_per_second,
        );
    }

    Ok(())
}
