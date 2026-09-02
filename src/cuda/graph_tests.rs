use std::time::Instant;

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::sys;
use half::bf16;

use crate::{ops, tensor::Shape};

use super::{
    CudaRuntime,
    benchmark::{BenchConfig, benchmark_gpu_paired},
};

const HIDDEN_SIZE: usize = 2048;
const CHAIN_REPETITIONS: usize = 32;
const RMS_EPS: f32 = 1.0e-5;

fn launch_decode_shaped_chain(
    runtime: &CudaRuntime,
    input: &crate::tensor::Tensor<bf16>,
    norm_weight: &crate::tensor::Tensor<bf16>,
    linear_weight: &crate::tensor::Tensor<bf16>,
    normalized: &mut crate::tensor::Tensor<bf16>,
    projected: &mut crate::tensor::Tensor<bf16>,
) -> Result<()> {
    for _ in 0..CHAIN_REPETITIONS {
        ops::rms_norm_bf16_into(runtime, input, norm_weight, RMS_EPS, normalized)?;
        ops::linear_bf16_into(runtime, normalized, linear_weight, projected)?;
    }
    Ok(())
}

fn mean_submit_us<F>(runtime: &CudaRuntime, iterations: usize, mut run: F) -> Result<f64>
where
    F: FnMut() -> Result<()>,
{
    runtime.synchronize()?;
    let started = Instant::now();
    for _ in 0..iterations {
        run()?;
    }
    let submit_us = started.elapsed().as_secs_f64() * 1_000_000.0 / iterations as f64;
    runtime.synchronize()?;
    Ok(submit_us)
}

#[test]
#[ignore = "RTX 5060 CUDA Graph launch-overhead probe"]
fn bench_cuda_graph_decode_shaped_launch_chain() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;

    let input_host = (0..HIDDEN_SIZE)
        .map(|index| {
            let value = ((index * 37 % 257) as f32 - 128.0) / 128.0;
            bf16::from_f32(value)
        })
        .collect::<Vec<_>>();
    let norm_host = vec![bf16::from_f32(1.0); HIDDEN_SIZE];
    let mut linear_host = vec![bf16::from_f32(0.0); HIDDEN_SIZE * HIDDEN_SIZE];
    for index in 0..HIDDEN_SIZE {
        linear_host[index * HIDDEN_SIZE + index] = bf16::from_f32(1.0);
    }

    let input = runtime.upload(&input_host, Shape::new([1, HIDDEN_SIZE]))?;
    let norm_weight = runtime.upload(&norm_host, Shape::new([HIDDEN_SIZE]))?;
    let linear_weight = runtime.upload(
        &linear_host,
        Shape::new([HIDDEN_SIZE, HIDDEN_SIZE]),
    )?;

    let mut direct_normalized = runtime.alloc_uninit::<bf16>(Shape::new([1, HIDDEN_SIZE]))?;
    let mut direct_projected = runtime.alloc_uninit::<bf16>(Shape::new([1, HIDDEN_SIZE]))?;
    let mut graph_normalized = runtime.alloc_uninit::<bf16>(Shape::new([1, HIDDEN_SIZE]))?;
    let mut graph_projected = runtime.alloc_uninit::<bf16>(Shape::new([1, HIDDEN_SIZE]))?;

    // Populate the cuBLASLt plan cache before stream capture. Phase 1 measures
    // replay of an already-warm persistent decode topology, not plan creation.
    launch_decode_shaped_chain(
        &runtime,
        &input,
        &norm_weight,
        &linear_weight,
        &mut direct_normalized,
        &mut direct_projected,
    )?;
    runtime.synchronize()?;

    let stream = runtime.stream();
    stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .context("failed to begin CUDA Graph stream capture")?;
    launch_decode_shaped_chain(
        &runtime,
        &input,
        &norm_weight,
        &linear_weight,
        &mut graph_normalized,
        &mut graph_projected,
    )?;
    let graph = stream
        .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD)
        .context("failed to end CUDA Graph stream capture")?
        .context("CUDA Graph capture returned no graph")?;
    runtime.synchronize()?;

    graph.launch().context("failed to launch captured CUDA Graph")?;
    runtime.synchronize()?;
    let direct_output = runtime.download(&direct_projected)?;
    let graph_output = runtime.download(&graph_projected)?;
    ensure!(
        direct_output == graph_output,
        "CUDA Graph probe changed BF16 output"
    );

    let stats = benchmark_gpu_paired(
        runtime.context(),
        runtime.stream(),
        BenchConfig {
            warmup: 10,
            batches: 30,
            iterations_per_batch: 20,
        },
        || {
            launch_decode_shaped_chain(
                &runtime,
                &input,
                &norm_weight,
                &linear_weight,
                &mut direct_normalized,
                &mut direct_projected,
            )
        },
        || {
            graph.launch().context("CUDA Graph replay failed")?;
            Ok(())
        },
    )?;

    let direct_submit_us = mean_submit_us(&runtime, 50, || {
        launch_decode_shaped_chain(
            &runtime,
            &input,
            &norm_weight,
            &linear_weight,
            &mut direct_normalized,
            &mut direct_projected,
        )
    })?;
    let graph_submit_us = mean_submit_us(&runtime, 50, || {
        graph.launch().context("CUDA Graph submit probe failed")?;
        Ok(())
    })?;

    println!(
        "cuda_graph_probe width={} chain_repetitions={} captured_launches={} direct_mean_us={:.3} direct_p50_us={:.3} direct_p95_us={:.3} graph_mean_us={:.3} graph_p50_us={:.3} graph_p95_us={:.3} speedup_mean={:.4}x speedup_p50={:.4}x speedup_p95={:.4}x speedup_min={:.4}x speedup_max={:.4}x direct_submit_us={:.3} graph_submit_us={:.3} submit_speedup={:.4}x exact_output=true",
        HIDDEN_SIZE,
        CHAIN_REPETITIONS,
        CHAIN_REPETITIONS * 2,
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
        direct_submit_us,
        graph_submit_us,
        direct_submit_us / graph_submit_us,
    );

    Ok(())
}
