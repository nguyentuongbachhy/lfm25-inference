use std::{env, fs, path::PathBuf};

use anyhow::{Context as _, Result, ensure};
use half::bf16;

use crate::{cache::KvPageSize, cuda::CudaRuntime};

use super::{Fp8PrecisionPolicy, Lfm2Model, RaggedBatchInput};

const CANDIDATE_LAYERS: [usize; 7] = [0, 1, 2, 3, 4, 5, 7];
const DECODE_STEPS: usize = 6;
const CACHE_DECODE_HEADROOM: usize = 32;

#[derive(Debug, Clone)]
struct Divergence {
    batch: usize,
    context: usize,
    step: usize,
    row: usize,
    baseline: u32,
    candidate: u32,
}

#[derive(Debug, Clone)]
struct MaskEvaluation {
    agreement: bool,
    max_nrmse: f64,
    min_cosine: f64,
    baseline_top1_margin: f64,
    first_divergence: Option<Divergence>,
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
) -> Result<super::BatchModelCache> {
    let page_size = KvPageSize::P16;
    let capacity = context + CACHE_DECODE_HEADROOM;
    let physical_pages = batch
        .checked_mul(capacity.div_ceil(page_size.value()))
        .context("INT8 sensitivity page count overflow")?;
    let maximum_batch_tokens = batch
        .checked_mul(context)
        .context("INT8 sensitivity prefill token count overflow")?;
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

fn row_metrics(reference: &[bf16], candidate: &[bf16]) -> (f64, f64) {
    let mut squared_error = 0.0f64;
    let mut reference_energy = 0.0f64;
    let mut candidate_energy = 0.0f64;
    let mut dot = 0.0f64;
    for (&reference, &candidate) in reference.iter().zip(candidate) {
        let reference = f64::from(reference.to_f32());
        let candidate = f64::from(candidate.to_f32());
        let delta = reference - candidate;
        squared_error += delta * delta;
        reference_energy += reference * reference;
        candidate_energy += candidate * candidate;
        dot += reference * candidate;
    }
    let nrmse = (squared_error / reference_energy.max(f64::MIN_POSITIVE)).sqrt();
    let cosine = dot
        / (reference_energy * candidate_energy)
            .sqrt()
            .max(f64::MIN_POSITIVE);
    (nrmse, cosine)
}

fn top1_margin(values: &[bf16]) -> f64 {
    let mut best = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for value in values {
        let value = value.to_f32();
        if value > best {
            second = best;
            best = value;
        } else if value > second {
            second = value;
        }
    }
    f64::from(best - second)
}

fn evaluate_mask(
    runtime: &CudaRuntime,
    model: &Lfm2Model,
    selected_layers: &[usize],
    cases: &[(usize, usize)],
    steps: usize,
) -> Result<MaskEvaluation> {
    let mut baseline = model.new_decode_executor_with_int8_tiny_m_down(runtime, 2, false)?;
    let mut candidate = model.new_decode_executor_with_int8_tiny_m_down(runtime, 2, false)?;
    candidate.set_int8_tiny_m_down_layers(runtime, model, selected_layers)?;

    let vocab = model.config().vocab_size;
    let mut agreement = true;
    let mut max_nrmse = 0.0f64;
    let mut min_cosine = 1.0f64;
    let mut baseline_top1_margin = f64::INFINITY;
    let mut first_divergence = None;

    for &(batch, context) in cases {
        let mut baseline_cache = prefill_cache(runtime, model, batch, context)?;
        let mut candidate_cache = prefill_cache(runtime, model, batch, context)?;
        let (request_slots, segment_offsets, segment_slots, output_rows) = decode_metadata(batch)?;

        for step in 0..steps {
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

            let baseline_tokens = {
                let sampled = model
                    .try_forward_ragged_decode(
                        runtime,
                        &mut baseline_cache,
                        &mut baseline,
                        &input,
                    )?
                    .context("baseline executor rejected INT8 sensitivity input")?;
                runtime.download(sampled)?
            };
            let baseline_logits = runtime.download(baseline.int8_test_logits())?;

            let candidate_tokens = {
                let sampled = model
                    .try_forward_ragged_decode(
                        runtime,
                        &mut candidate_cache,
                        &mut candidate,
                        &input,
                    )?
                    .context("candidate executor rejected INT8 sensitivity input")?;
                runtime.download(sampled)?
            };
            let candidate_logits = runtime.download(candidate.int8_test_logits())?;

            ensure!(
                baseline_logits.len() == batch * vocab && candidate_logits.len() == batch * vocab,
                "INT8 sensitivity logits shape mismatch"
            );
            for row in 0..batch {
                let reference = &baseline_logits[row * vocab..(row + 1) * vocab];
                let candidate_row = &candidate_logits[row * vocab..(row + 1) * vocab];
                let (nrmse, cosine) = row_metrics(reference, candidate_row);
                max_nrmse = max_nrmse.max(nrmse);
                min_cosine = min_cosine.min(cosine);
                baseline_top1_margin = baseline_top1_margin.min(top1_margin(reference));
                if baseline_tokens[row] != candidate_tokens[row] {
                    agreement = false;
                    if first_divergence.is_none() {
                        first_divergence = Some(Divergence {
                            batch,
                            context,
                            step,
                            row,
                            baseline: baseline_tokens[row],
                            candidate: candidate_tokens[row],
                        });
                    }
                }
            }
        }
    }

    Ok(MaskEvaluation {
        agreement,
        max_nrmse,
        min_cosine,
        baseline_top1_margin,
        first_divergence,
    })
}

#[test]
#[ignore = "real checkpoint INT8 layer-sensitivity diagnostic"]
fn diagnose_int8_tiny_m_down_layer_sensitivity() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let model = load_model(&runtime)?;
    let failing_case = [(1usize, 128usize)];
    let full_grid = [
        (1usize, 128usize),
        (2, 128),
        (1, 512),
        (2, 512),
        (1, 2048),
        (2, 2048),
    ];

    let mut ranked = Vec::with_capacity(CANDIDATE_LAYERS.len());
    for layer in CANDIDATE_LAYERS {
        let evaluation = evaluate_mask(&runtime, &model, &[layer], &failing_case, 1)?;
        println!(
            "int8_layer_sensitivity layer={} agreement={} max_nrmse={:.8} min_cosine={:.8} baseline_top1_margin={:.6} divergence={:?}",
            layer,
            evaluation.agreement,
            evaluation.max_nrmse,
            evaluation.min_cosine,
            evaluation.baseline_top1_margin,
            evaluation.first_divergence,
        );
        ranked.push((layer, evaluation.max_nrmse));
    }
    ranked.sort_by(|left, right| left.1.total_cmp(&right.1));
    println!(
        "int8_layer_sensitivity_ranked={:?}",
        ranked.iter().map(|(layer, _)| *layer).collect::<Vec<_>>()
    );

    let mut selected = Vec::new();
    for (layer, _) in ranked {
        let mut trial = selected.clone();
        trial.push(layer);
        trial.sort_unstable();
        let evaluation = evaluate_mask(&runtime, &model, &trial, &full_grid, DECODE_STEPS)?;
        println!(
            "int8_mask_trial add_layer={} mask={:?} agreement={} max_nrmse={:.8} min_cosine={:.8} min_baseline_top1_margin={:.6} divergence={:?}",
            layer,
            trial,
            evaluation.agreement,
            evaluation.max_nrmse,
            evaluation.min_cosine,
            evaluation.baseline_top1_margin,
            evaluation.first_divergence,
        );
        if evaluation.agreement {
            selected = trial;
        }
    }

    println!("int8_recommended_mask={selected:?}");
    Ok(())
}
