use std::{env, fs, path::PathBuf};

use anyhow::{Context as _, Result, ensure};

use crate::{cache::KvPageSize, cuda::CudaRuntime};

use super::{Fp8PrecisionPolicy, Lfm2Model, RaggedBatchInput};

const DECODE_STEPS: usize = 6;
const BENCH_WARMUP_STEPS: usize = 4;
const BENCH_MEASURED_STEPS: usize = 20;
const CACHE_DECODE_HEADROOM: usize = 64;

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
) -> Result<super::BatchModelCache> {
    let page_size = KvPageSize::P16;
    let capacity = context + CACHE_DECODE_HEADROOM;
    let physical_pages = batch
        .checked_mul(capacity.div_ceil(page_size.value()))
        .context("INT8 production test page count overflow")?;
    let maximum_batch_tokens = batch
        .checked_mul(context)
        .context("INT8 production test prefill token count overflow")?;
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

#[test]
#[ignore = "real checkpoint production INT8 validation"]
fn production_int8_tiny_m_matches_baseline_argmax_for_forced_history() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let model = load_model(&runtime)?;

    for context in [128usize, 512, 2048] {
        for batch in [1usize, 2] {
            let mut baseline_cache = prefill_cache(&runtime, &model, batch, context)?;
            let mut candidate_cache = prefill_cache(&runtime, &model, batch, context)?;
            let mut baseline =
                model.new_decode_executor_with_int8_tiny_m_down(&runtime, batch, false)?;
            let mut candidate =
                model.new_decode_executor_with_int8_tiny_m_down(&runtime, batch, true)?;
            let (request_slots, segment_offsets, segment_slots, output_rows) =
                decode_metadata(batch)?;

            for step in 0..DECODE_STEPS {
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

                let baseline_sampled = model
                    .try_forward_ragged_decode(
                        &runtime,
                        &mut baseline_cache,
                        &mut baseline,
                        &input,
                    )?
                    .context("baseline production executor rejected INT8 validation input")?;
                let baseline_tokens = runtime.download(baseline_sampled)?;

                let candidate_sampled = model
                    .try_forward_ragged_decode(
                        &runtime,
                        &mut candidate_cache,
                        &mut candidate,
                        &input,
                    )?
                    .context("candidate production executor rejected INT8 validation input")?;
                let candidate_tokens = runtime.download(candidate_sampled)?;

                ensure!(
                    baseline_tokens == candidate_tokens,
                    "INT8 production argmax diverged at B={batch} context={context} step={step}: baseline={baseline_tokens:?} candidate={candidate_tokens:?}"
                );
            }
            eprintln!(
                "INT8 production forced-history agreement passed B={batch} context={context} steps={DECODE_STEPS}"
            );
        }
    }
    Ok(())
}

#[derive(Debug)]
struct DecodePass {
    samples_ms: Vec<f64>,
    final_sampled: Vec<u32>,
}

fn run_decode_pass(
    runtime: &CudaRuntime,
    model: &Lfm2Model,
    batch: usize,
    context: usize,
    int8_enabled: bool,
) -> Result<DecodePass> {
    let mut cache = prefill_cache(runtime, model, batch, context)?;
    let mut executor =
        model.new_decode_executor_with_int8_tiny_m_down(runtime, batch, int8_enabled)?;
    let (request_slots, segment_offsets, segment_slots, output_rows) = decode_metadata(batch)?;

    for step in 0..BENCH_WARMUP_STEPS {
        let token_ids = forced_tokens(batch, step)?;
        let positions = vec![u32::try_from(context + step)?; batch];
        let sampled = model
            .try_forward_ragged_decode(
                runtime,
                &mut cache,
                &mut executor,
                &RaggedBatchInput {
                    token_ids: &token_ids,
                    positions: &positions,
                    request_slots: &request_slots,
                    segment_offsets: &segment_offsets,
                    segment_slots: &segment_slots,
                    output_rows: &output_rows,
                },
            )?
            .context("production executor rejected INT8 benchmark warmup input")?;
        let _ = sampled;
    }
    runtime.synchronize()?;

    let mut samples_ms = Vec::with_capacity(BENCH_MEASURED_STEPS);
    let mut final_sampled = Vec::new();
    for measured in 0..BENCH_MEASURED_STEPS {
        let step = BENCH_WARMUP_STEPS + measured;
        let token_ids = forced_tokens(batch, step)?;
        let positions = vec![u32::try_from(context + step)?; batch];
        let started = runtime.record_timing_event()?;
        let sampled = model
            .try_forward_ragged_decode(
                runtime,
                &mut cache,
                &mut executor,
                &RaggedBatchInput {
                    token_ids: &token_ids,
                    positions: &positions,
                    request_slots: &request_slots,
                    segment_offsets: &segment_offsets,
                    segment_slots: &segment_slots,
                    output_rows: &output_rows,
                },
            )?
            .context("production executor rejected INT8 benchmark input")?;
        let finished = runtime.record_timing_event()?;
        samples_ms.push(runtime.elapsed_ms(&started, &finished)?);
        final_sampled = runtime.download(sampled)?;
    }

    Ok(DecodePass {
        samples_ms,
        final_sampled,
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
#[ignore = "real checkpoint production INT8 ABBA benchmark"]
fn bench_production_int8_tiny_m_down_abba() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let model = load_model(&runtime)?;

    for context in [128usize, 512, 2048] {
        for batch in [1usize, 2] {
            let mut baseline_samples = Vec::with_capacity(BENCH_MEASURED_STEPS * 2);
            let mut candidate_samples = Vec::with_capacity(BENCH_MEASURED_STEPS * 2);
            let mut reference_tokens: Option<Vec<u32>> = None;
            let mut top1_agreement = true;

            // Balanced AB/BA across whole production passes. Each pass starts
            // from the same prefill state and receives the same forced history.
            for int8_enabled in [false, true, true, false] {
                let pass = run_decode_pass(&runtime, &model, batch, context, int8_enabled)?;
                match &reference_tokens {
                    Some(reference) => top1_agreement &= pass.final_sampled == *reference,
                    None => reference_tokens = Some(pass.final_sampled.clone()),
                }
                if int8_enabled {
                    candidate_samples.extend(pass.samples_ms);
                } else {
                    baseline_samples.extend(pass.samples_ms);
                }
            }

            let baseline_mean = mean(&baseline_samples);
            let candidate_mean = mean(&candidate_samples);
            let baseline_p95 = percentile(&baseline_samples, 0.95);
            let candidate_p95 = percentile(&candidate_samples, 0.95);
            println!(
                "int8_production_down B={} C={} baseline_mean_ms={:.6} int8_mean_ms={:.6} mean_speedup={:.4}x baseline_p95_ms={:.6} int8_p95_ms={:.6} p95_speedup={:.4}x top1_agreement={}",
                batch,
                context,
                baseline_mean,
                candidate_mean,
                baseline_mean / candidate_mean,
                baseline_p95,
                candidate_p95,
                baseline_p95 / candidate_p95,
                top1_agreement,
            );
            ensure!(top1_agreement, "INT8 production ABBA final argmax mismatch at B={batch} C={context}");
        }
    }
    Ok(())
}
