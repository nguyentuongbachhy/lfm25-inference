use std::time::Instant;

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, sys};
use half::bf16;

use super::{
    benchmark::{BenchConfig, benchmark_gpu_paired},
    blaslt::BlasLt,
};

const HIDDEN_SIZE: usize = 2048;
const CHAIN_REPETITIONS: usize = 32;

fn launch_decode_shaped_chain(
    blaslt: &BlasLt,
    input: &CudaSlice<bf16>,
    linear_weight: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
) -> Result<()> {
    for _ in 0..CHAIN_REPETITIONS {
        unsafe {
            blaslt.linear_bf16(input, linear_weight, output, 1, HIDDEN_SIZE, HIDDEN_SIZE)?;
        }
    }
    Ok(())
}

fn mean_submit_us<F>(stream: &CudaStream, iterations: usize, mut run: F) -> Result<f64>
where
    F: FnMut() -> Result<()>,
{
    stream.synchronize()?;
    let started = Instant::now();
    for _ in 0..iterations {
        run()?;
    }
    let submit_us = started.elapsed().as_secs_f64() * 1_000_000.0 / iterations as f64;
    stream.synchronize()?;
    Ok(submit_us)
}

#[test]
#[ignore = "RTX 5060 CUDA Graph launch-overhead probe"]
fn bench_cuda_graph_decode_shaped_launch_chain() -> Result<()> {
    let context = CudaContext::new(0).context("failed to create CUDA graph probe context")?;

    // CudaRuntime uses a non-default stream. cudarc then enables per-allocation
    // event tracking so DevicePtr/DevicePtrMut can synchronize cross-stream use.
    // During stream capture, waits on events recorded before capture become a
    // dependency across the capture boundary and CUDA reports
    // CUDA_ERROR_STREAM_CAPTURE_ISOLATION.
    //
    // This probe owns exactly one stream and explicitly synchronizes it. Disable
    // event tracking before the stream and every device allocation are created.
    // Production CudaRuntime synchronization behavior is not changed.
    unsafe {
        context.disable_event_tracking();
    }
    let stream = context
        .new_stream()
        .context("failed to create CUDA graph probe stream")?;
    let blaslt = BlasLt::new(stream.clone()).context("failed to create graph probe cuBLASLt")?;
    blaslt
        .prepare_linear_bf16(1, HIDDEN_SIZE, HIDDEN_SIZE)
        .context("failed to prepare graph probe cuBLASLt plan")?;

    let input_host = (0..HIDDEN_SIZE)
        .map(|index| {
            let value = ((index * 37 % 257) as f32 - 128.0) / 128.0;
            bf16::from_f32(value)
        })
        .collect::<Vec<_>>();
    let mut linear_host = vec![bf16::from_f32(0.0); HIDDEN_SIZE * HIDDEN_SIZE];
    for index in 0..HIDDEN_SIZE {
        linear_host[index * HIDDEN_SIZE + index] = bf16::from_f32(1.0);
    }

    let input = stream
        .clone_htod(&input_host)
        .context("failed to upload graph probe input")?;
    let linear_weight = stream
        .clone_htod(&linear_host)
        .context("failed to upload graph probe weight")?;
    let mut direct_output = stream
        .alloc_zeros::<bf16>(HIDDEN_SIZE)
        .context("failed to allocate direct graph probe output")?;
    let mut graph_output = stream
        .alloc_zeros::<bf16>(HIDDEN_SIZE)
        .context("failed to allocate captured graph probe output")?;

    // Warm the selected cuBLASLt plan and lazy driver/library state before
    // capture. Phase 1 measures replay, not plan or module creation.
    launch_decode_shaped_chain(&blaslt, &input, &linear_weight, &mut direct_output)?;
    stream.synchronize()?;

    stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .context("failed to begin CUDA Graph stream capture")?;
    launch_decode_shaped_chain(&blaslt, &input, &linear_weight, &mut graph_output)?;
    let graph = stream
        .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY)
        .context("failed to end CUDA Graph stream capture")?
        .context("CUDA Graph capture returned no graph")?;
    graph.upload().context("failed to upload CUDA Graph")?;
    stream.synchronize()?;

    graph
        .launch()
        .context("failed to launch captured CUDA Graph")?;
    stream.synchronize()?;
    let direct_host = stream
        .clone_dtoh(&direct_output)
        .context("failed to download direct graph probe output")?;
    let graph_host = stream
        .clone_dtoh(&graph_output)
        .context("failed to download captured graph probe output")?;
    ensure!(
        direct_host == graph_host,
        "CUDA Graph probe changed BF16 output"
    );

    let stats = benchmark_gpu_paired(
        &context,
        &stream,
        BenchConfig {
            warmup: 10,
            batches: 30,
            iterations_per_batch: 20,
        },
        || launch_decode_shaped_chain(&blaslt, &input, &linear_weight, &mut direct_output),
        || {
            graph.launch().context("CUDA Graph replay failed")?;
            Ok(())
        },
    )?;

    let direct_submit_us = mean_submit_us(&stream, 50, || {
        launch_decode_shaped_chain(&blaslt, &input, &linear_weight, &mut direct_output)
    })?;
    let graph_submit_us = mean_submit_us(&stream, 50, || {
        graph.launch().context("CUDA Graph submit probe failed")?;
        Ok(())
    })?;

    println!(
        "cuda_graph_probe width={} chain_repetitions={} captured_launches={} direct_mean_us={:.3} direct_p50_us={:.3} direct_p95_us={:.3} graph_mean_us={:.3} graph_p50_us={:.3} graph_p95_us={:.3} speedup_mean={:.4}x speedup_p50={:.4}x speedup_p95={:.4}x speedup_min={:.4}x speedup_max={:.4}x direct_submit_us={:.3} graph_submit_us={:.3} submit_speedup={:.4}x exact_output=true",
        HIDDEN_SIZE,
        CHAIN_REPETITIONS,
        CHAIN_REPETITIONS,
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
