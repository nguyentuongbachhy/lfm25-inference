use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cuda::{
        CudaRuntime,
        benchmark::{BenchConfig, benchmark_gpu_paired},
        testing::readback,
    },
    tensor::Shape,
};

use super::{residual_rms_norm_bf16_into, residual_rms_norm_bf16_to_e4m3_into};

const HIDDEN: usize = 2048;
const QUANT_SCALE: f32 = 153.3262;

fn values(elements: usize, mul: usize, modulus: usize, center: f32, scale: f32) -> Vec<bf16> {
    (0..elements)
        .map(|index| bf16::from_f32(((index * mul % modulus) as f32 - center) / scale))
        .collect()
}

#[test]
#[ignore = "GPU residual RMSNorm plus E4M3 fusion benchmark"]
fn bench_residual_rms_norm_fp8_fusion() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let config = BenchConfig {
        warmup: 20,
        batches: 50,
        iterations_per_batch: 50,
    };

    for rows in [1usize, 8, 16] {
        let residual = runtime.upload(
            &values(rows * HIDDEN, 17, 257, 128.0, 64.0),
            Shape::new([rows, HIDDEN]),
        )?;
        let update = runtime.upload(
            &values(rows * HIDDEN, 23, 251, 125.0, 96.0),
            Shape::new([rows, HIDDEN]),
        )?;
        let weight = runtime.upload(
            &(0..HIDDEN)
                .map(|index| bf16::from_f32(0.75 + (index * 7 % 31) as f32 / 64.0))
                .collect::<Vec<_>>(),
            Shape::new([HIDDEN]),
        )?;

        let mut reference_sum = runtime.alloc_bf16(Shape::new([rows, HIDDEN]))?;
        let mut reference_norm = runtime.alloc_bf16(Shape::new([rows, HIDDEN]))?;
        let mut reference_fp8 = runtime.alloc_fp8(Shape::new([rows, HIDDEN]))?;
        let mut candidate_sum = runtime.alloc_bf16(Shape::new([rows, HIDDEN]))?;
        let mut candidate_fp8 = runtime.alloc_fp8(Shape::new([rows, HIDDEN]))?;

        let mut run_reference = || -> Result<()> {
            residual_rms_norm_bf16_into(
                &runtime,
                &residual,
                &update,
                &weight,
                1e-5,
                &mut reference_sum,
                &mut reference_norm,
            )?;
            unsafe {
                runtime.kernels().fp8_quantize().launch_bf16_e4m3(
                    runtime.stream(),
                    reference_norm.storage(),
                    reference_fp8.storage_mut(),
                    reference_norm.numel(),
                    QUANT_SCALE,
                )?;
            }
            Ok(())
        };
        let mut run_candidate = || -> Result<()> {
            residual_rms_norm_bf16_to_e4m3_into(
                &runtime,
                &residual,
                &update,
                &weight,
                1e-5,
                QUANT_SCALE,
                &mut candidate_sum,
                &mut candidate_fp8,
            )
        };

        run_reference()?;
        run_candidate()?;
        runtime.synchronize()?;
        let residual_exact = readback(&runtime, &reference_sum)? == readback(&runtime, &candidate_sum)?;
        let fp8_exact = readback(&runtime, &reference_fp8)? == readback(&runtime, &candidate_fp8)?;
        ensure!(residual_exact, "fused residual output mismatch at M={rows}");
        ensure!(fp8_exact, "fused FP8 output mismatch at M={rows}");

        let stats = benchmark_gpu_paired(
            runtime.context(),
            runtime.stream(),
            config,
            &mut run_reference,
            &mut run_candidate,
        )?;
        println!(
            "rms_fp8_fusion M={} reference_mean_us={:.3} reference_p50_us={:.3} reference_p95_us={:.3} candidate_mean_us={:.3} candidate_p50_us={:.3} candidate_p95_us={:.3} speedup_mean={:.4}x speedup_p50={:.4}x speedup_p95={:.4}x speedup_min={:.4}x speedup_max={:.4}x residual_exact={} fp8_exact={}",
            rows,
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
            residual_exact,
            fp8_exact,
        );
    }
    Ok(())
}
