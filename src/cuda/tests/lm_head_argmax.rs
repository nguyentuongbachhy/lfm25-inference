use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cuda::{
        CudaRuntime,
        benchmark::{BenchConfig, benchmark_gpu},
        blaslt::{Fp8LinearConfig, fp8::Fp8ScaleMode},
    },
    ops::argmax_rows_bf16_atomic_into,
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
#[ignore = "GPU benchmark: production-shape FP8 LM head plus atomic greedy argmax"]
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

    // A uniform non-zero E4M3 matrix keeps the allocation/setup simple while
    // preserving the exact production M/N/K geometry, dtype, weight traffic and
    // cuBLASLt algorithm. Numerical values do not affect the primitive timing
    // question in this phase.
    let weight_host = vec![0x38u8; N * K]; // E4M3 +1.0
    let weight = runtime.upload(&weight_host, Shape::new([N, K]))?;
    drop(weight_host);

    let mut hidden_fp8 = runtime.zeros::<u8>(Shape::new([M, K]))?;
    let mut logits = runtime.zeros::<bf16>(Shape::new([M, N]))?;
    let mut sampled = runtime.alloc_u32(Shape::new([M]))?;

    runtime
        .blaslt()
        .prepare_linear_fp8(M, N, K, Fp8ScaleMode::Tensorwide)?;

    // Populate the FP8 input and logits once before measuring isolated stages.
    unsafe {
        runtime.kernels().fp8_quantize().launch_bf16_e4m3(
            runtime.stream(),
            hidden.storage(),
            hidden_fp8.storage_mut(),
            hidden.numel(),
            1.0,
        )?;
        runtime.blaslt().linear_fp8(
            hidden_fp8.storage(),
            weight.storage(),
            logits.storage_mut(),
            fp8_config(),
        )?;
    }
    argmax_rows_bf16_atomic_into(&runtime, &logits, &mut sampled)?;
    runtime.synchronize()?;
    let token_a = runtime.download(&sampled)?[0];

    // Repeat once outside timing to require deterministic greedy output.
    unsafe {
        runtime.kernels().fp8_quantize().launch_bf16_e4m3(
            runtime.stream(),
            hidden.storage(),
            hidden_fp8.storage_mut(),
            hidden.numel(),
            1.0,
        )?;
        runtime.blaslt().linear_fp8(
            hidden_fp8.storage(),
            weight.storage(),
            logits.storage_mut(),
            fp8_config(),
        )?;
    }
    argmax_rows_bf16_atomic_into(&runtime, &logits, &mut sampled)?;
    runtime.synchronize()?;
    let token_b = runtime.download(&sampled)?[0];
    ensure!(token_a == token_b, "LM-head boundary token is not deterministic");

    let quantize = benchmark_gpu(runtime.context(), runtime.stream(), config, || unsafe {
        runtime.kernels().fp8_quantize().launch_bf16_e4m3(
            runtime.stream(),
            hidden.storage(),
            hidden_fp8.storage_mut(),
            hidden.numel(),
            1.0,
        )
    })?;

    let lm_head = benchmark_gpu(runtime.context(), runtime.stream(), config, || unsafe {
        runtime.blaslt().linear_fp8(
            hidden_fp8.storage(),
            weight.storage(),
            logits.storage_mut(),
            fp8_config(),
        )
    })?;

    let argmax = benchmark_gpu(runtime.context(), runtime.stream(), config, || {
        argmax_rows_bf16_atomic_into(&runtime, &logits, &mut sampled)
    })?;

    let boundary = benchmark_gpu(runtime.context(), runtime.stream(), config, || {
        unsafe {
            runtime.kernels().fp8_quantize().launch_bf16_e4m3(
                runtime.stream(),
                hidden.storage(),
                hidden_fp8.storage_mut(),
                hidden.numel(),
                1.0,
            )?;
            runtime.blaslt().linear_fp8(
                hidden_fp8.storage(),
                weight.storage(),
                logits.storage_mut(),
                fp8_config(),
            )?;
        }
        argmax_rows_bf16_atomic_into(&runtime, &logits, &mut sampled)
    })?;

    let removable_us = boundary.mean_us - lm_head.mean_us;
    let fusion_ceiling = boundary.mean_us / lm_head.mean_us;
    let boundary_fraction_6ms = boundary.mean_us / 6_000.0;

    println!(
        "lm_head_argmax_phase0 quantize_mean_us={:.3} quantize_p95_us={:.3} lm_head_mean_us={:.3} lm_head_p95_us={:.3} argmax_mean_us={:.3} argmax_p95_us={:.3} boundary_mean_us={:.3} boundary_p95_us={:.3} removable_us={:.3} fusion_ceiling={:.4}x boundary_fraction_6ms={:.4} deterministic_token={}",
        quantize.mean_us,
        quantize.p95_us,
        lm_head.mean_us,
        lm_head.p95_us,
        argmax.mean_us,
        argmax.p95_us,
        boundary.mean_us,
        boundary.p95_us,
        removable_us,
        fusion_ceiling,
        boundary_fraction_6ms,
        token_a,
    );

    Ok(())
}
