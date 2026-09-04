use std::{env, fs, path::PathBuf, time::Instant};

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::sys;

use crate::{cache::KvPageSize, cuda::CudaRuntime, model::Fp8PrecisionPolicy, ops};

use super::*;

const BENCH_WARMUP_STEPS: usize = 4;
const BENCH_MEASURED_STEPS: usize = 16;
const CACHE_DECODE_HEADROOM: usize = 64;
// B1/C4096 is reproducibly positive while B1/C5120 is already negative.
// All points below remain in the same PS16 Split-K=8 topology, so this sweep
// narrows only the context-dependent graph crossover without reopening Split-K.
const BENCH_SHAPES: &[(usize, usize)] = &[(1, 4096), (1, 4352), (1, 4608), (1, 4864), (1, 5120)];

fn model_dir() -> PathBuf {
    env::var_os("LFM25_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/LFM2.5-1.2B-Instruct"))
}

fn policy_path() -> PathBuf {
    env::var_os("LFM25_FP8_POLICY")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs/benchmarks/fp8/selected-policy.json"))
}

fn load_model(runtime: &CudaRuntime) -> Result<Lfm2Model> {
    let mut model = Lfm2Model::load(runtime, &model_dir())?;
    let bytes = fs::read(policy_path()).context("failed to read selected FP8 policy")?;
    let policy: Fp8PrecisionPolicy =
        serde_json::from_slice(&bytes).context("failed to parse selected FP8 policy")?;
    let enabled = model.install_fp8_policy(runtime, &policy)?;
    ensure!(enabled > 0, "selected FP8 policy enables no sites");
    Ok(model)
}

fn prefill_cache(
    runtime: &CudaRuntime,
    model: &Lfm2Model,
    batch: usize,
    context: usize,
) -> Result<BatchModelCache> {
    let page_size = KvPageSize::P16;
    let capacity = context + CACHE_DECODE_HEADROOM;
    let physical_pages = batch
        .checked_mul(capacity.div_ceil(page_size.value()))
        .context("CUDA Graph benchmark page count overflow")?;
    let maximum_batch_tokens = batch
        .checked_mul(context)
        .context("CUDA Graph benchmark prefill token count overflow")?;
    let mut cache = model.new_batch_cache(
        runtime,
        batch,
        maximum_batch_tokens.max(batch),
        physical_pages,
        page_size,
    )?;
    for slot in 0..batch {
        cache.reserve(slot, capacity)?;
    }

    let mut token_ids = Vec::with_capacity(maximum_batch_tokens);
    let mut positions = Vec::with_capacity(maximum_batch_tokens);
    let mut request_slots = Vec::with_capacity(maximum_batch_tokens);
    let mut segment_offsets = Vec::with_capacity(batch + 1);
    let mut segment_slots = Vec::with_capacity(batch);
    let mut output_rows = Vec::with_capacity(batch);
    segment_offsets.push(0);

    for slot in 0..batch {
        for position in 0..context {
            let token = if position == 0 {
                model.config().bos_token_id
            } else {
                100u32 + u32::try_from((slot * 37 + position * 13) % 4000)?
            };
            token_ids.push(token);
            positions.push(u32::try_from(position)?);
            request_slots.push(u32::try_from(slot)?);
        }
        segment_offsets.push(u32::try_from(token_ids.len())?);
        segment_slots.push(u32::try_from(slot)?);
        output_rows.push(u32::try_from(token_ids.len() - 1)?);
    }

    let _ = model.forward_ragged_batch(
        runtime,
        &mut cache,
        RaggedBatchInput {
            token_ids: &token_ids,
            positions: &positions,
            request_slots: &request_slots,
            segment_offsets: &segment_offsets,
            segment_slots: &segment_slots,
            output_rows: &output_rows,
        },
    )?;
    Ok(cache)
}

#[allow(clippy::type_complexity)]
fn decode_metadata(batch: usize) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>)> {
    let request_slots = (0..batch)
        .map(u32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let segment_offsets = (0..=batch)
        .map(u32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let segment_slots = request_slots.clone();
    let output_rows = request_slots.clone();
    Ok((request_slots, segment_offsets, segment_slots, output_rows))
}

fn forced_tokens(batch: usize, step: usize) -> Result<Vec<u32>> {
    (0..batch)
        .map(|slot| Ok(5000u32 + u32::try_from(step * 17 + slot)?))
        .collect::<Result<Vec<_>>>()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DecodeMode {
    Direct,
    Graph,
}

struct DecodePass {
    gpu_samples_ms: Vec<f64>,
    submit_samples_us: Vec<f64>,
    sampled_trace: Vec<Vec<u32>>,
}

fn topology_signature(batch: usize, context_tokens: usize) -> (bool, usize) {
    let mok = ops::should_use_mok_one_kernel(16, context_tokens, batch);
    let splits = if mok {
        0
    } else {
        ops::splitk_decode_splits(batch, context_tokens, 16)
    };
    (mok, splits)
}

fn ensure_stable_topology(batch: usize, context: usize, total_steps: usize) -> Result<()> {
    let captured = topology_signature(batch, context + 1);
    for step in 1..total_steps {
        let current = topology_signature(batch, context + step + 1);
        ensure!(
            current == captured,
            "CUDA Graph benchmark crosses a decode topology boundary: B={batch} C={context} capture={captured:?} step={step} current={current:?}"
        );
    }
    Ok(())
}

fn warm_decode_path(runtime: &CudaRuntime, model: &Lfm2Model) -> Result<()> {
    let batch = 1;
    let context = 128;
    let mut cache = prefill_cache(runtime, model, batch, context)?;
    let mut executor = model.new_decode_executor(runtime, batch)?;
    let (request_slots, segment_offsets, segment_slots, output_rows) = decode_metadata(batch)?;
    let token_ids = forced_tokens(batch, 0)?;
    let positions = vec![u32::try_from(context)?; batch];
    let input = RaggedBatchInput {
        token_ids: &token_ids,
        positions: &positions,
        request_slots: &request_slots,
        segment_offsets: &segment_offsets,
        segment_slots: &segment_slots,
        output_rows: &output_rows,
    };
    ensure!(
        executor.eligible(&input),
        "warm decode input is not graph eligible"
    );
    cache.prepare_ragged(runtime, &input)?;
    executor.forward_prepared(model, runtime, &mut cache)?;
    runtime.synchronize()
}

fn run_decode_pass(
    runtime: &CudaRuntime,
    model: &Lfm2Model,
    batch: usize,
    context: usize,
    mode: DecodeMode,
) -> Result<DecodePass> {
    let total_steps = 1 + BENCH_WARMUP_STEPS + BENCH_MEASURED_STEPS;
    ensure_stable_topology(batch, context, total_steps)?;

    let mut cache = prefill_cache(runtime, model, batch, context)?;
    let mut executor = model.new_decode_executor(runtime, batch)?;
    let (request_slots, segment_offsets, segment_slots, output_rows) = decode_metadata(batch)?;
    let mut gpu_samples_ms = Vec::with_capacity(BENCH_MEASURED_STEPS);
    let mut submit_samples_us = Vec::with_capacity(BENCH_MEASURED_STEPS);
    let mut sampled_trace = Vec::with_capacity(total_steps);
    let mut graph = None;

    for step in 0..total_steps {
        let token_ids = forced_tokens(batch, step)?;
        let positions = vec![u32::try_from(context + step)?; batch];
        let input = RaggedBatchInput {
            token_ids: &token_ids,
            positions: &positions,
            request_slots: &request_slots,
            segment_offsets: &segment_offsets,
            segment_slots: &segment_slots,
            output_rows: &output_rows,
        };
        ensure!(
            executor.eligible(&input),
            "decode executor rejected CUDA Graph benchmark input"
        );

        if mode == DecodeMode::Graph && step == 0 {
            cache.prepare_ragged(runtime, &input)?;
            runtime.synchronize()?;
            let stream = runtime.stream();
            stream
                .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .context("failed to begin full-model CUDA Graph capture")?;
            executor.forward_prepared(model, runtime, &mut cache)?;
            let captured = stream
                .end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY,
                )
                .context("failed to end full-model CUDA Graph capture")?
                .context("full-model CUDA Graph capture returned no graph")?;
            captured
                .upload()
                .context("failed to pre-upload full-model CUDA Graph")?;
            runtime.synchronize()?;
            captured
                .launch()
                .context("failed to execute captured full-model step")?;
            runtime.synchronize()?;
            sampled_trace.push(runtime.download(&executor.sampled)?);
            graph = Some(captured);
            continue;
        }

        let gpu_started = runtime.record_timing_event()?;
        let submit_started = Instant::now();
        cache.prepare_ragged(runtime, &input)?;
        match mode {
            DecodeMode::Direct => executor.forward_prepared(model, runtime, &mut cache)?,
            DecodeMode::Graph => graph
                .as_ref()
                .context("full-model CUDA Graph was not captured")?
                .launch()
                .context("full-model CUDA Graph replay failed")?,
        }
        let gpu_finished = runtime.record_timing_event()?;
        let submit_us = submit_started.elapsed().as_secs_f64() * 1_000_000.0;
        let gpu_ms = runtime.elapsed_ms(&gpu_started, &gpu_finished)?;
        sampled_trace.push(runtime.download(&executor.sampled)?);

        if step > BENCH_WARMUP_STEPS {
            gpu_samples_ms.push(gpu_ms);
            submit_samples_us.push(submit_us);
        }
    }

    ensure!(
        gpu_samples_ms.len() == BENCH_MEASURED_STEPS
            && submit_samples_us.len() == BENCH_MEASURED_STEPS,
        "CUDA Graph benchmark sample count mismatch"
    );
    Ok(DecodePass {
        gpu_samples_ms,
        submit_samples_us,
        sampled_trace,
    })
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    sorted[((sorted.len() - 1) as f64 * quantile).round() as usize]
}

#[test]
#[ignore = "real-checkpoint full-model CUDA Graph ABBA benchmark"]
fn bench_cuda_graph_full_model_abba() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;

    unsafe {
        runtime.context().disable_event_tracking();
    }

    let model = load_model(&runtime)?;
    warm_decode_path(&runtime, &model)?;

    for &(batch, context) in BENCH_SHAPES {
        let mut direct_gpu = Vec::with_capacity(BENCH_MEASURED_STEPS * 2);
        let mut graph_gpu = Vec::with_capacity(BENCH_MEASURED_STEPS * 2);
        let mut direct_submit = Vec::with_capacity(BENCH_MEASURED_STEPS * 2);
        let mut graph_submit = Vec::with_capacity(BENCH_MEASURED_STEPS * 2);
        let mut reference_trace: Option<Vec<Vec<u32>>> = None;
        let mut top1_agreement = true;

        for mode in [
            DecodeMode::Direct,
            DecodeMode::Graph,
            DecodeMode::Graph,
            DecodeMode::Direct,
        ] {
            let pass = run_decode_pass(&runtime, &model, batch, context, mode)?;
            match &reference_trace {
                Some(reference) => top1_agreement &= pass.sampled_trace == *reference,
                None => reference_trace = Some(pass.sampled_trace.clone()),
            }
            match mode {
                DecodeMode::Direct => {
                    direct_gpu.extend(pass.gpu_samples_ms);
                    direct_submit.extend(pass.submit_samples_us);
                }
                DecodeMode::Graph => {
                    graph_gpu.extend(pass.gpu_samples_ms);
                    graph_submit.extend(pass.submit_samples_us);
                }
            }
        }

        ensure!(
            top1_agreement,
            "CUDA Graph sampled-token trace mismatch at B={batch} C={context}"
        );

        let direct_mean = mean(&direct_gpu);
        let graph_mean = mean(&graph_gpu);
        let direct_p95 = percentile(&direct_gpu, 0.95);
        let graph_p95 = percentile(&graph_gpu, 0.95);
        let direct_submit_mean = mean(&direct_submit);
        let graph_submit_mean = mean(&graph_submit);
        println!(
            "cuda_graph_full_model B={} C={} direct_mean_ms={:.6} graph_mean_ms={:.6} mean_speedup={:.4}x direct_p95_ms={:.6} graph_p95_ms={:.6} p95_speedup={:.4}x direct_submit_us={:.3} graph_submit_us={:.3} submit_speedup={:.4}x top1_agreement={}",
            batch,
            context,
            direct_mean,
            graph_mean,
            direct_mean / graph_mean,
            direct_p95,
            graph_p95,
            direct_p95 / graph_p95,
            direct_submit_mean,
            graph_submit_mean,
            direct_submit_mean / graph_submit_mean,
            top1_agreement,
        );
    }

    Ok(())
}
