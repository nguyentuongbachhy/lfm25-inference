use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cuda::{
        CudaRuntime,
        benchmark::{BenchConfig, benchmark_gpu, benchmark_gpu_paired},
        blaslt::{Fp8LinearConfig, fp8::Fp8ScaleMode},
    },
    ops::argmax_rows_bf16_into,
    tensor::Shape,
};

const M: usize = 1;
const K: usize = 2_048;
const N: usize = 65_536;

fn fp8_config() -> Fp8LinearConfig {
    Fp8LinearConfig {
        m: M,
        n: N,
        k: K,
        scale_mode: Fp8ScaleMode::Tensorwide,
        output_scale: 1.0,
    }
}

#[test]
#[ignore = "GPU benchmark: production-shape FP8 LM head plus production greedy argmax dispatch"]
fn bench_fp8_lm_head_argmax_boundary() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let config = BenchConfig {
        warmup: 20,
        batches: 40,
        iterations_per_batch: 5,
    };

    let hidden_host = (0..K)
        .map(|index| {
            let value = match index & 3 {
                0 => 0.5,
                1 => -0.5,
                2 => 0.25,
                _ => -0.25,
            };
            bf16::from_f32(value)
        })
        .collect::<Vec<_>>();
    let hidden = runtime.upload(&hidden_host, Shape::new([M, K]))?;

    // Non-zero E4M3 data preserves the exact production M/N/K geometry, dtype,
    // weight traffic and cuBLASLt plan while keeping setup deterministic.
    let weight_host = vec![0x38u8; N * K]; // E4M3 +1.0
    let weight = runtime.upload(&weight_host, Shape::new([N, K]))?;
    drop(weight_host);

    // Keep reference and boundary outputs separate. benchmark_gpu_paired owns
    // both closures at the same time, so separate buffers also make the compared
    // paths independent and avoid overlapping mutable borrows.
    let mut lm_hidden_fp8 = runtime.zeros::<u8>(Shape::new([M, K]))?;
    let mut lm_logits = runtime.zeros::<bf16>(Shape::new([M, N]))?;
    let mut boundary_hidden_fp8 = runtime.zeros::<u8>(Shape::new([M, K]))?;
    let mut boundary_logits = runtime.zeros::<bf16>(Shape::new([M, N]))?;
    let mut boundary_sampled = runtime.alloc_u32(Shape::new([M]))?;

    runtime
        .blaslt()
        .prepare_linear_fp8(M, N, K, Fp8ScaleMode::Tensorwide)?;

    // Prepare the LM-head-only lower-bound input once. The complete boundary
    // re-quantizes its own input on every measured iteration.
    unsafe {
        runtime.kernels().fp8_quantize().launch_bf16_e4m3(
            runtime.stream(),
            hidden.storage(),
            lm_hidden_fp8.storage_mut(),
            hidden.numel(),
            1.0,
        )?;
    }

    // Require deterministic production greedy output before timing.
    for _ in 0..2 {
        unsafe {
            runtime.kernels().fp8_quantize().launch_bf16_e4m3(
                runtime.stream(),
                hidden.storage(),
                boundary_hidden_fp8.storage_mut(),
                hidden.numel(),
                1.0,
            )?;
            runtime.blaslt().linear_fp8(
                boundary_hidden_fp8.storage(),
                weight.storage(),
                boundary_logits.storage_mut(),
                fp8_config(),
            )?;
        }
        argmax_rows_bf16_into(&runtime, &boundary_logits, &mut boundary_sampled)?;
        runtime.synchronize()?;
    }
    let token_a = runtime.download(&boundary_sampled)?[0];

    unsafe {
        runtime.kernels().fp8_quantize().launch_bf16_e4m3(
            runtime.stream(),
            hidden.storage(),
            boundary_hidden_fp8.storage_mut(),
            hidden.numel(),
            1.0,
        )?;
        runtime.blaslt().linear_fp8(
            boundary_hidden_fp8.storage(),
            weight.storage(),
            boundary_logits.storage_mut(),
            fp8_config(),
        )?;
    }
    argmax_rows_bf16_into(&runtime, &boundary_logits, &mut boundary_sampled)?;
    runtime.synchronize()?;
    let token_b = runtime.download(&boundary_sampled)?[0];
    ensure!(token_a == token_b, "LM-head boundary token is not deterministic");

    // Directly pair the complete current boundary against an optimistic lower
    // bound that retains only the existing cuBLASLt LM-head projection. This is
    // the decision measurement. AB/BA alternation removes the sequential
    // thermal/clock bias seen in the first Phase-0 run.
    let paired = benchmark_gpu_paired(
        runtime.context(),
        runtime.stream(),
        config,
        || {
            unsafe {
                runtime.kernels().fp8_quantize().launch_bf16_e4m3(
                    runtime.stream(),
                    hidden.storage(),
                    boundary_hidden_fp8.storage_mut(),
                    hidden.numel(),
                    1.0,
                )?;
                runtime.blaslt().linear_fp8(
                    boundary_hidden_fp8.storage(),
                    weight.storage(),
                    boundary_logits.storage_mut(),
                    fp8_config(),
                )?;
            }
            argmax_rows_bf16_into(&runtime, &boundary_logits, &mut boundary_sampled)
        },
        || unsafe {
            runtime.blaslt().linear_fp8(
                lm_hidden_fp8.storage(),
                weight.storage(),
                lm_logits.storage_mut(),
                fp8_config(),
            )
        },
    )?;

    // Keep isolated stages only as diagnostics. The continuation decision uses
    // the paired complete-boundary versus LM-head-only result above.
    let quantize = benchmark_gpu(runtime.context(), runtime.stream(), config, || unsafe {
        runtime.kernels().fp8_quantize().launch_bf16_e4m3(
            runtime.stream(),
            hidden.storage(),
            boundary_hidden_fp8.storage_mut(),
            hidden.numel(),
            1.0,
        )
    })?;
    let argmax = benchmark_gpu(runtime.context(), runtime.stream(), config, || {
        argmax_rows_bf16_into(&runtime, &boundary_logits, &mut boundary_sampled)
    })?;

    let boundary = &paired.reference;
    let lm_head = &paired.candidate;
    let removable_us = boundary.mean_us - lm_head.mean_us;
    let fusion_ceiling = boundary.mean_us / lm_head.mean_us;
    let boundary_fraction_6ms = boundary.mean_us / 6_000.0;

    println!(
        "lm_head_argmax_phase0_paired boundary_mean_us={:.3} boundary_p95_us={:.3} lm_head_mean_us={:.3} lm_head_p95_us={:.3} fusion_ceiling={:.4}x paired_speedup_mean={:.4}x paired_speedup_p50={:.4}x paired_speedup_p95={:.4}x removable_us={:.3} quantize_mean_us={:.3} quantize_p95_us={:.3} argmax_mean_us={:.3} argmax_p95_us={:.3} boundary_fraction_6ms={:.4} deterministic_token={}",
        boundary.mean_us,
        boundary.p95_us,
        lm_head.mean_us,
        lm_head.p95_us,
        fusion_ceiling,
        paired.speedup_mean,
        paired.speedup_p50,
        paired.speedup_p95,
        removable_us,
        quantize.mean_us,
        quantize.p95_us,
        argmax.mean_us,
        argmax.p95_us,
        boundary_fraction_6ms,
        token_a,
    );

    Ok(())
}
