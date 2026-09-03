use std::{env, fs, path::PathBuf, time::Instant};

use anyhow::{Context as _, Result, ensure};

use crate::{cache::KvPageSize, cuda::CudaRuntime, model::Fp8PrecisionPolicy};

use super::*;

const WARMUP_STEPS: usize = 4;
const MEASURED_STEPS: usize = 16;
const CACHE_HEADROOM: usize = 64;
const SHAPES: &[(usize, usize)] = &[
    (1, 128),
    (16, 128),
    (1, 2048),
    (8, 2048),
    (1, 8192),
];

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
    ensure!(
        model.install_fp8_policy(runtime, &policy)? > 0,
        "selected FP8 policy enables no sites"
    );
    ensure!(
        model.weights.layers.iter().any(|layer| layer.feed_forward.gate_up.fp8.is_some()),
        "selected FP8 policy has no Gate/Up sites for RMS FP8 fusion"
    );
    ensure!(
        model.weights.lm_head_fp8.is_some(),
        "selected FP8 policy has no FP8 LM head"
    );
    Ok(model)
}

fn prefill_cache(
    runtime: &CudaRuntime,
    model: &Lfm2Model,
    batch: usize,
    context: usize,
) -> Result<BatchModelCache> {
    let page_size = KvPageSize::P16;
    let capacity = context + CACHE_HEADROOM;
    let physical_pages = batch
        .checked_mul(capacity.div_ceil(page_size.value()))
        .context("RMS FP8 benchmark page count overflow")?;
    let maximum_batch_tokens = batch
        .checked_mul(context)
        .context("RMS FP8 benchmark token count overflow")?;
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
            token_ids.push(if position == 0 {
                model.config().bos_token_id
            } else {
                100u32 + u32::try_from((slot * 37 + position * 13) % 4000)?
            });
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

struct FusionPass {
    gpu_ms: Vec<f64>,
    submit_us: Vec<f64>,
    trace: Vec<Vec<u32>>,
}

fn run_pass(
    runtime: &CudaRuntime,
    model: &Lfm2Model,
    batch: usize,
    context: usize,
    fusion_enabled: bool,
) -> Result<FusionPass> {
    let mut cache = prefill_cache(runtime, model, batch, context)?;
    let mut executor = model.new_decode_executor(runtime, batch)?;
    executor.set_cuda_graphs_enabled_for_test(false);
    executor.set_rms_fp8_fusion_enabled_for_test(fusion_enabled);

    let request_slots = (0..batch)
        .map(u32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let segment_offsets = (0..=batch)
        .map(u32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let segment_slots = request_slots.clone();
    let output_rows = request_slots.clone();
    let total_steps = WARMUP_STEPS + MEASURED_STEPS;
    let mut gpu_ms = Vec::with_capacity(MEASURED_STEPS);
    let mut submit_us = Vec::with_capacity(MEASURED_STEPS);
    let mut trace = Vec::with_capacity(total_steps);

    for step in 0..total_steps {
        let token_ids = (0..batch)
            .map(|slot| Ok(5000u32 + u32::try_from(step * 17 + slot)?))
            .collect::<Result<Vec<_>>>()?;
        let positions = vec![u32::try_from(context + step)?; batch];
        let input = RaggedBatchInput {
            token_ids: &token_ids,
            positions: &positions,
            request_slots: &request_slots,
            segment_offsets: &segment_offsets,
            segment_slots: &segment_slots,
            output_rows: &output_rows,
        };

        let gpu_started = runtime.record_timing_event()?;
        let submit_started = Instant::now();
        let sampled = model
            .try_forward_ragged_decode(runtime, &mut cache, &mut executor, &input)?
            .context("RMS FP8 production decode unexpectedly fell back")?;
        let gpu_finished = runtime.record_timing_event()?;
        let submit = submit_started.elapsed().as_secs_f64() * 1_000_000.0;
        let elapsed = runtime.elapsed_ms(&gpu_started, &gpu_finished)?;
        trace.push(runtime.download(sampled)?);

        if step >= WARMUP_STEPS {
            gpu_ms.push(elapsed);
            submit_us.push(submit);
        }
    }

    Ok(FusionPass {
        gpu_ms,
        submit_us,
        trace,
    })
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn percentile(values: &[f64], q: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

#[test]
#[ignore = "real-checkpoint residual RMSNorm FP8 full-model ABBA benchmark"]
fn bench_rms_fp8_fusion_full_model_abba() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    ensure!(
        !runtime.graph_capture_compatible(),
        "unset LFM25_CUDA_GRAPHS for the RMS FP8 fusion isolation benchmark"
    );
    let model = load_model(&runtime)?;

    for &(batch, context) in SHAPES {
        let mut direct_gpu = Vec::with_capacity(MEASURED_STEPS * 2);
        let mut fused_gpu = Vec::with_capacity(MEASURED_STEPS * 2);
        let mut direct_submit = Vec::with_capacity(MEASURED_STEPS * 2);
        let mut fused_submit = Vec::with_capacity(MEASURED_STEPS * 2);
        let mut reference_trace: Option<Vec<Vec<u32>>> = None;
        let mut top1_agreement = true;

        for fusion_enabled in [false, true, true, false] {
            let pass = run_pass(&runtime, &model, batch, context, fusion_enabled)?;
            match &reference_trace {
                Some(reference) => top1_agreement &= pass.trace == *reference,
                None => reference_trace = Some(pass.trace.clone()),
            }
            if fusion_enabled {
                fused_gpu.extend(pass.gpu_ms);
                fused_submit.extend(pass.submit_us);
            } else {
                direct_gpu.extend(pass.gpu_ms);
                direct_submit.extend(pass.submit_us);
            }
        }

        ensure!(
            top1_agreement,
            "RMS FP8 fusion token trace mismatch at B={batch} C={context}"
        );
        let direct_mean = mean(&direct_gpu);
        let fused_mean = mean(&fused_gpu);
        let direct_p95 = percentile(&direct_gpu, 0.95);
        let fused_p95 = percentile(&fused_gpu, 0.95);
        let direct_submit_mean = mean(&direct_submit);
        let fused_submit_mean = mean(&fused_submit);
        println!(
            "rms_fp8_full_model B={} C={} direct_mean_ms={:.6} fused_mean_ms={:.6} mean_speedup={:.4}x direct_p95_ms={:.6} fused_p95_ms={:.6} p95_speedup={:.4}x direct_submit_us={:.3} fused_submit_us={:.3} submit_speedup={:.4}x top1_agreement={}",
            batch,
            context,
            direct_mean,
            fused_mean,
            direct_mean / fused_mean,
            direct_p95,
            fused_p95,
            direct_submit_mean,
            fused_submit_mean,
            direct_submit_mean / fused_submit_mean,
            top1_agreement,
        );
    }
    Ok(())
}
