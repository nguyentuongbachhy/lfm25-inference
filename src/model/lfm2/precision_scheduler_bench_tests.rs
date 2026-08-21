use std::{env, fs, path::PathBuf};

use anyhow::{Context as _, Result, ensure};

use crate::{
    cache::KvPageSize,
    cuda::CudaRuntime,
    model::Fp8PrecisionPolicy,
    ops,
};

use super::{DecodeExecutor, Lfm2Model, RaggedBatchInput};

#[derive(Debug)]
struct DecodePass {
    samples_ms: Vec<f64>,
    final_sampled: Vec<u32>,
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

fn load_model_with_policy(runtime: &CudaRuntime) -> Result<Lfm2Model> {
    let mut model = Lfm2Model::load(runtime, &model_dir())?;
    let bytes = fs::read(policy_path()).context("failed to read selected FP8 policy")?;
    let policy: Fp8PrecisionPolicy =
        serde_json::from_slice(&bytes).context("failed to parse selected FP8 policy")?;
    let enabled = model.install_fp8_policy(runtime, &policy)?;
    ensure!(enabled > 0, "selected FP8 policy enables no sites");
    Ok(model)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    sorted[((sorted.len() - 1) as f64 * quantile).round() as usize]
}

fn run_production_decode_pass(
    model: &mut Lfm2Model,
    runtime: &CudaRuntime,
    executor: &mut DecodeExecutor,
    batch: usize,
    context: usize,
    use_fp8: bool,
    warmup_steps: usize,
    measured_steps: usize,
) -> Result<DecodePass> {
    ensure!(batch > 0 && context > 0, "invalid decode profile shape");
    ensure!(measured_steps > 0, "decode profile requires samples");
    model.set_decode_fp8_enabled(use_fp8)?;

    let page_size = KvPageSize::P16;
    let capacity = context
        .checked_add(warmup_steps)
        .and_then(|value| value.checked_add(measured_steps))
        .context("decode profile capacity overflow")?;
    let physical_pages = batch
        .checked_mul(capacity.div_ceil(page_size.value()))
        .context("decode profile page count overflow")?;
    let mut cache = model.new_batch_cache(runtime, batch, batch, physical_pages, page_size)?;
    for slot in 0..batch {
        cache.reserve(slot, capacity)?;
    }
    cache.prime_context(runtime, batch, context)?;

    let request_slots = (0..batch)
        .map(u32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let segment_offsets = (0..=batch)
        .map(u32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let segment_slots = request_slots.clone();
    let output_rows = request_slots.clone();
    let mut token_ids = vec![42u32; batch];
    let mut positions = vec![u32::try_from(context)?; batch];

    for step in 0..warmup_steps {
        token_ids.fill(42 + u32::try_from(step % 17)?);
        positions.fill(u32::try_from(context + step)?);
        let sampled = model
            .try_forward_ragged_decode(
                runtime,
                &mut cache,
                executor,
                &RaggedBatchInput {
                    token_ids: &token_ids,
                    positions: &positions,
                    request_slots: &request_slots,
                    segment_offsets: &segment_offsets,
                    segment_slots: &segment_slots,
                    output_rows: &output_rows,
                },
            )?
            .context("production decode executor rejected benchmark batch")?;
        let _ = sampled;
    }
    runtime.synchronize()?;

    let mut samples_ms = Vec::with_capacity(measured_steps);
    let mut final_sampled = Vec::new();
    for step in 0..measured_steps {
        token_ids.fill(59 + u32::try_from(step % 23)?);
        positions.fill(u32::try_from(context + warmup_steps + step)?);
        let started = runtime.record_timing_event()?;
        let sampled = model
            .try_forward_ragged_decode(
                runtime,
                &mut cache,
                executor,
                &RaggedBatchInput {
                    token_ids: &token_ids,
                    positions: &positions,
                    request_slots: &request_slots,
                    segment_offsets: &segment_offsets,
                    segment_slots: &segment_slots,
                    output_rows: &output_rows,
                },
            )?
            .context("production decode executor rejected measured batch")?;
        let finished = runtime.record_timing_event()?;
        samples_ms.push(runtime.elapsed_ms(&started, &finished)?);
        if step + 1 == measured_steps {
            final_sampled = runtime.download(sampled)?;
        }
    }

    ensure!(
        final_sampled.len() == batch,
        "production decode sample count mismatch"
    );
    ensure!(
        final_sampled.windows(2).all(|window| window[0] == window[1]),
        "identical production decode rows produced different top-1 tokens"
    );
    for slot in 0..batch {
        cache.release(runtime, slot)?;
    }
    Ok(DecodePass {
        samples_ms,
        final_sampled,
    })
}

fn run_legacy_decode_top1(
    model: &mut Lfm2Model,
    runtime: &CudaRuntime,
    batch: usize,
    context: usize,
    use_fp8: bool,
) -> Result<Vec<u32>> {
    model.set_decode_fp8_enabled(use_fp8)?;
    let page_size = KvPageSize::P16;
    let capacity = context.checked_add(1).context("legacy capacity overflow")?;
    let physical_pages = batch
        .checked_mul(capacity.div_ceil(page_size.value()))
        .context("legacy page count overflow")?;
    let mut cache = model.new_batch_cache(runtime, batch, batch, physical_pages, page_size)?;
    for slot in 0..batch {
        cache.reserve(slot, capacity)?;
    }
    cache.prime_context(runtime, batch, context)?;
    let request_slots = (0..batch)
        .map(u32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let token_ids = vec![73u32; batch];
    let positions = vec![u32::try_from(context)?; batch];
    let logits = model.forward_decode_batch(
        runtime,
        &mut cache,
        &token_ids,
        &positions,
        &request_slots,
    )?;
    let sampled = ops::argmax_rows_bf16(runtime, &logits)?;
    let host = runtime.download(&sampled)?;
    for slot in 0..batch {
        cache.release(runtime, slot)?;
    }
    Ok(host)
}

fn run_executor_decode_top1(
    model: &mut Lfm2Model,
    runtime: &CudaRuntime,
    executor: &mut DecodeExecutor,
    batch: usize,
    context: usize,
    use_fp8: bool,
) -> Result<Vec<u32>> {
    model.set_decode_fp8_enabled(use_fp8)?;
    let page_size = KvPageSize::P16;
    let capacity = context.checked_add(1).context("executor capacity overflow")?;
    let physical_pages = batch
        .checked_mul(capacity.div_ceil(page_size.value()))
        .context("executor page count overflow")?;
    let mut cache = model.new_batch_cache(runtime, batch, batch, physical_pages, page_size)?;
    for slot in 0..batch {
        cache.reserve(slot, capacity)?;
    }
    cache.prime_context(runtime, batch, context)?;
    let request_slots = (0..batch)
        .map(u32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let segment_offsets = (0..=batch)
        .map(u32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let token_ids = vec![73u32; batch];
    let positions = vec![u32::try_from(context)?; batch];
    let sampled = model
        .try_forward_ragged_decode(
            runtime,
            &mut cache,
            executor,
            &RaggedBatchInput {
                token_ids: &token_ids,
                positions: &positions,
                request_slots: &request_slots,
                segment_offsets: &segment_offsets,
                segment_slots: &request_slots,
                output_rows: &request_slots,
            },
        )?
        .context("production decode executor rejected correctness batch")?;
    let host = runtime.download(sampled)?;
    for slot in 0..batch {
        cache.release(runtime, slot)?;
    }
    Ok(host)
}

#[test]
#[ignore = "requires CUDA, model weights, and selected FP8 policy"]
fn production_decode_executor_matches_legacy_top1_bf16_and_fp8() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let mut model = load_model_with_policy(&runtime)?;
    model.prepare_batched_fp8(&runtime, 4)?;
    model.set_decode_fp8_enabled(true)?;
    let mut executor = model.new_decode_executor(&runtime, 4)?;

    for use_fp8 in [false, true] {
        for context in [128usize, 512, 2048] {
            let legacy = run_legacy_decode_top1(&mut model, &runtime, 4, context, use_fp8)?;
            let production =
                run_executor_decode_top1(&mut model, &runtime, &mut executor, 4, context, use_fp8)?;
            assert_eq!(
                production, legacy,
                "production executor top-1 mismatch at precision={} context={context}",
                if use_fp8 { "fp8" } else { "bf16" }
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "GPU benchmark requiring model weights and selected FP8 policy"]
fn bench_production_decode_precision_grid_abba() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let mut model = load_model_with_policy(&runtime)?;
    model.set_decode_fp8_enabled(true)?;
    let maximum_batch = 64usize;
    let mut executor = model.new_decode_executor(&runtime, maximum_batch)?;
    let warmup_steps = 4usize;
    let measured_steps = 20usize;
    let page_size = KvPageSize::P16;
    let attention_layers = model
        .config()
        .layer_types
        .iter()
        .filter(|kind| kind.as_str() == "full_attention")
        .count();
    let bytes_per_page = attention_layers
        .checked_mul(2)
        .and_then(|value| value.checked_mul(model.config().num_key_value_heads))
        .and_then(|value| value.checked_mul(page_size.value()))
        .and_then(|value| value.checked_mul(model.config().head_dim()))
        .and_then(|value| value.checked_mul(std::mem::size_of::<half::bf16>()))
        .context("KV page byte size overflow")?;

    for context in [128usize, 512, 2048, 8192] {
        for batch in [1usize, 2, 4, 8, 16, 32, 64] {
            let capacity = context + warmup_steps + measured_steps;
            let pages = batch
                .checked_mul(capacity.div_ceil(page_size.value()))
                .context("profile page count overflow")?;
            let required_kv_bytes = pages
                .checked_mul(bytes_per_page)
                .context("profile KV byte requirement overflow")?;
            let (free_vram_bytes, _) = runtime.memory_info()?;
            if required_kv_bytes >= free_vram_bytes {
                println!(
                    "production_decode_precision_skip batch={batch} context={context} required_kv_bytes={required_kv_bytes} free_vram_bytes={free_vram_bytes}"
                );
                continue;
            }

            let bf16_first = run_production_decode_pass(
                &mut model,
                &runtime,
                &mut executor,
                batch,
                context,
                false,
                warmup_steps,
                measured_steps,
            )?;
            let fp8_first = run_production_decode_pass(
                &mut model,
                &runtime,
                &mut executor,
                batch,
                context,
                true,
                warmup_steps,
                measured_steps,
            )?;
            let fp8_second = run_production_decode_pass(
                &mut model,
                &runtime,
                &mut executor,
                batch,
                context,
                true,
                warmup_steps,
                measured_steps,
            )?;
            let bf16_second = run_production_decode_pass(
                &mut model,
                &runtime,
                &mut executor,
                batch,
                context,
                false,
                warmup_steps,
                measured_steps,
            )?;

            let mut bf16_samples = bf16_first.samples_ms;
            bf16_samples.extend(bf16_second.samples_ms);
            let mut fp8_samples = fp8_first.samples_ms;
            fp8_samples.extend(fp8_second.samples_ms);
            let top1_agreement = bf16_first.final_sampled == fp8_first.final_sampled
                && bf16_first.final_sampled == fp8_second.final_sampled
                && bf16_first.final_sampled == bf16_second.final_sampled;
            let bf16_mean = mean(&bf16_samples);
            let bf16_p50 = percentile(&bf16_samples, 0.50);
            let bf16_p95 = percentile(&bf16_samples, 0.95);
            let fp8_mean = mean(&fp8_samples);
            let fp8_p50 = percentile(&fp8_samples, 0.50);
            let fp8_p95 = percentile(&fp8_samples, 0.95);
            println!(
                "production_decode_precision batch={batch} context={context} bf16_mean_ms={bf16_mean:.6} bf16_p50_ms={bf16_p50:.6} bf16_p95_ms={bf16_p95:.6} fp8_mean_ms={fp8_mean:.6} fp8_p50_ms={fp8_p50:.6} fp8_p95_ms={fp8_p95:.6} speedup_mean={:.4}x speedup_p95={:.4}x top1_agreement={top1_agreement}",
                bf16_mean / fp8_mean,
                bf16_p95 / fp8_p95,
            );
        }
    }
    model.set_decode_fp8_enabled(true)?;
    Ok(())
}
