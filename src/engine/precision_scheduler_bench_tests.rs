use std::{env, path::PathBuf, time::Instant};

use anyhow::{Context as _, Result, ensure};

use crate::{
    cache::KvPageSize,
    model::RaggedBatchInput,
};

use super::{Engine, EngineConfig};

fn model_dir() -> PathBuf {
    env::var_os("LFM25_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/LFM2.5-1.2B-Instruct"))
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    sorted[((sorted.len() - 1) as f64 * quantile).round() as usize]
}

#[derive(Debug)]
struct PrefillSamples {
    wall_ms: Vec<f64>,
    gpu_ms: Vec<f64>,
    submit_ms: Vec<f64>,
}

fn benchmark_prefill_tokens(
    engine: &Engine,
    tokens: usize,
    warmup_steps: usize,
    measured_steps: usize,
) -> Result<PrefillSamples> {
    ensure!(tokens > 0, "prefill token count must be positive");
    ensure!(measured_steps > 0, "prefill benchmark requires samples");

    let run_once = || -> Result<(f64, f64, f64)> {
        let pages = tokens.div_ceil(engine.config.kv_page_size.value());
        let mut cache = engine.model.new_batch_cache(
            &engine.runtime,
            1,
            tokens,
            pages,
            engine.config.kv_page_size,
        )?;
        cache.reserve(0, tokens)?;

        let mut token_ids = vec![42u32; tokens];
        token_ids[0] = engine.model.config().bos_token_id;
        let positions = (0..tokens)
            .map(u32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let request_slots = vec![0u32; tokens];
        let segment_offsets = [0u32, u32::try_from(tokens)?];
        let segment_slots = [0u32];
        let output_rows = [u32::try_from(tokens - 1)?];

        let gpu_started = engine.runtime.record_timing_event()?;
        let wall_started = Instant::now();
        let submit_started = Instant::now();
        let logits = engine.model.forward_ragged_batch(
            &engine.runtime,
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
        let gpu_finished = engine.runtime.record_timing_event()?;
        let submit_ms = submit_started.elapsed().as_secs_f64() * 1000.0;
        let gpu_ms = engine.runtime.elapsed_ms(&gpu_started, &gpu_finished)?;
        let wall_ms = wall_started.elapsed().as_secs_f64() * 1000.0;
        drop(logits);
        Ok((wall_ms, gpu_ms, submit_ms))
    };

    for _ in 0..warmup_steps {
        let _ = run_once()?;
    }
    engine.runtime.synchronize()?;

    let mut wall_ms = Vec::with_capacity(measured_steps);
    let mut gpu_ms = Vec::with_capacity(measured_steps);
    let mut submit_ms = Vec::with_capacity(measured_steps);
    for _ in 0..measured_steps {
        let (wall, gpu, submit) = run_once()?;
        wall_ms.push(wall);
        gpu_ms.push(gpu);
        submit_ms.push(submit);
    }

    Ok(PrefillSamples {
        wall_ms,
        gpu_ms,
        submit_ms,
    })
}

#[test]
#[ignore = "GPU benchmark requiring model weights"]
fn bench_precision_scheduler_dense_prefill_grid() -> Result<()> {
    let engine = Engine::load(
        &model_dir(),
        0,
        EngineConfig {
            kv_page_size: KvPageSize::P16,
            ..EngineConfig::default()
        },
    )?;

    for tokens in [
        16usize, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512,
    ] {
        let samples = benchmark_prefill_tokens(&engine, tokens, 2, 9)
            .with_context(|| format!("failed dense prefill benchmark at T={tokens}"))?;
        println!(
            "scheduler_prefill_profile tokens={tokens} wall_mean_ms={:.6} wall_p50_ms={:.6} wall_p95_ms={:.6} gpu_mean_ms={:.6} gpu_p95_ms={:.6} submit_mean_ms={:.6}",
            mean(&samples.wall_ms),
            percentile(&samples.wall_ms, 0.50),
            percentile(&samples.wall_ms, 0.95),
            mean(&samples.gpu_ms),
            percentile(&samples.gpu_ms, 0.95),
            mean(&samples.submit_ms),
        );
    }
    Ok(())
}