use std::{env, fs, path::PathBuf, time::Instant};

use anyhow::{Context as _, Result, ensure};

use crate::{
    cache::KvPageSize,
    cuda::CudaRuntime,
    model::{Fp8PrecisionPolicy, Lfm2Model, RaggedBatchInput},
};

const BENCH_WARMUP_STEPS: usize = 4;
const BENCH_MEASURED_STEPS: usize = 12;
const CACHE_DECODE_HEADROOM: usize = 64;
const BENCH_SHAPES: &[(usize, usize)] = &[
    (1, 128),
    (16, 128),
    (1, 2048),
    (8, 2048),
    (1, 8192),
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum PackedMode {
    Direct,
    Packed,
}

struct DecodePass {
    gpu_samples_ms: Vec<f64>,
    submit_samples_us: Vec<f64>,
    sampled_trace: Vec<Vec<u32>>,
}

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
) -> Result<crate::model::BatchModelCache> {
    let page_size = KvPageSize::P16;
    let capacity = context + CACHE_DECODE_HEADROOM;
    let physical_pages = batch
        .checked_mul(capacity.div_ceil(page_size.value()))
        .context("packed QKV benchmark page count overflow")?;
    let maximum_batch_tokens = batch
        .checked_mul(context)
        .context("packed QKV benchmark prefill token count overflow")?;
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

fn set_packed_mode(mode: PackedMode) {
    // This ignored benchmark is required to run with --test-threads=1. Rust
    // 2024 makes process-environment mutation unsafe because other threads can
    // race reads; the single-thread test contract makes the mutation bounded.
    unsafe {
        env::set_var(
            "LFM25_PACKED_QKV",
            if mode == PackedMode::Packed { "1" } else { "0" },
        );
    }
}

fn run_decode_pass(
    runtime: &CudaRuntime,
    model: &Lfm2Model,
    batch: usize,
    context: usize,
    mode: PackedMode,
) -> Result<DecodePass> {
    set_packed_mode(mode);
    let mut cache = prefill_cache(runtime, model, batch, context)?;
    let mut executor = model.new_decode_executor(runtime, batch)?;
    let (request_slots, segment_offsets, segment_slots, output_rows) = decode_metadata(batch)?;
    let total_steps = BENCH_WARMUP_STEPS + BENCH_MEASURED_STEPS;
    let mut gpu_samples_ms = Vec::with_capacity(BENCH_MEASURED_STEPS);
    let mut submit_samples_us = Vec::with_capacity(BENCH_MEASURED_STEPS);
    let mut sampled_trace = Vec::with_capacity(total_steps);

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
        let gpu_started = runtime.record_timing_event()?;
        let submit_started = Instant::now();
        let sampled = model
            .try_forward_ragged_decode(runtime, &mut cache, &mut executor, &input)?
            .context("packed QKV full-model input was not decode-executor eligible")?;
        let gpu_finished = runtime.record_timing_event()?;
        let submit_us = submit_started.elapsed().as_secs_f64() * 1_000_000.0;
        let gpu_ms = runtime.elapsed_ms(&gpu_started, &gpu_finished)?;
        sampled_trace.push(runtime.download(sampled)?);

        if step >= BENCH_WARMUP_STEPS {
            gpu_samples_ms.push(gpu_ms);
            submit_samples_us.push(submit_us);
        }
    }

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
#[ignore = "real-checkpoint packed QKV full-model ABBA benchmark"]
fn bench_packed_qkv_full_model_abba() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let model = load_model(&runtime)?;

    for &(batch, context) in BENCH_SHAPES {
        let mut direct_gpu = Vec::with_capacity(BENCH_MEASURED_STEPS * 2);
        let mut packed_gpu = Vec::with_capacity(BENCH_MEASURED_STEPS * 2);
        let mut direct_submit = Vec::with_capacity(BENCH_MEASURED_STEPS * 2);
        let mut packed_submit = Vec::with_capacity(BENCH_MEASURED_STEPS * 2);
        let mut reference_trace: Option<Vec<Vec<u32>>> = None;
        let mut top1_agreement = true;
        let mut top1_matches = 0usize;
        let mut top1_total = 0usize;
        let mut first_divergence: Option<(usize, usize, u32, u32)> = None;

        for mode in [
            PackedMode::Direct,
            PackedMode::Packed,
            PackedMode::Packed,
            PackedMode::Direct,
        ] {
            let pass = run_decode_pass(&runtime, &model, batch, context, mode)?;
            match &reference_trace {
                Some(reference) => {
                    top1_agreement &= pass.sampled_trace == *reference;
                    for (step, (expected_step, actual_step)) in
                        reference.iter().zip(&pass.sampled_trace).enumerate()
                    {
                        for (slot, (&expected, &actual)) in
                            expected_step.iter().zip(actual_step).enumerate()
                        {
                            top1_total += 1;
                            if expected == actual {
                                top1_matches += 1;
                            } else if first_divergence.is_none() {
                                first_divergence = Some((step, slot, expected, actual));
                            }
                        }
                    }
                }
                None => reference_trace = Some(pass.sampled_trace.clone()),
            }
            match mode {
                PackedMode::Direct => {
                    direct_gpu.extend(pass.gpu_samples_ms);
                    direct_submit.extend(pass.submit_samples_us);
                }
                PackedMode::Packed => {
                    packed_gpu.extend(pass.gpu_samples_ms);
                    packed_submit.extend(pass.submit_samples_us);
                }
            }
        }

        let direct_mean = mean(&direct_gpu);
        let packed_mean = mean(&packed_gpu);
        let direct_p95 = percentile(&direct_gpu, 0.95);
        let packed_p95 = percentile(&packed_gpu, 0.95);
        let direct_submit_mean = mean(&direct_submit);
        let packed_submit_mean = mean(&packed_submit);
        let top1_match_ratio = if top1_total == 0 {
            1.0
        } else {
            top1_matches as f64 / top1_total as f64
        };
        println!(
            "packed_qkv_full_model B={} C={} direct_mean_ms={:.6} packed_mean_ms={:.6} mean_speedup={:.4}x direct_p95_ms={:.6} packed_p95_ms={:.6} p95_speedup={:.4}x direct_submit_us={:.3} packed_submit_us={:.3} submit_speedup={:.4}x top1_agreement={} top1_match_ratio={:.6} first_divergence={:?}",
            batch,
            context,
            direct_mean,
            packed_mean,
            direct_mean / packed_mean,
            direct_p95,
            packed_p95,
            direct_p95 / packed_p95,
            direct_submit_mean,
            packed_submit_mean,
            direct_submit_mean / packed_submit_mean,
            top1_agreement,
            top1_match_ratio,
            first_divergence,
        );
    }

    unsafe {
        env::remove_var("LFM25_PACKED_QKV");
    }
    Ok(())
}
