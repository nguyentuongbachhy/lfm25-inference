use std::{env, path::PathBuf};

use anyhow::{Context as _, Result, ensure};
use half::bf16;

use crate::{
    cache::KvPageSize,
    cuda::{
        CudaRuntime,
        benchmark::{BenchConfig, benchmark_gpu_paired},
        testing::readback,
    },
    model::{CalibrationCollector, CalibrationPhase, Lfm2Model},
    ops::{
        Int8TinyMWorkspace, linear_bf16_into, linear_int8_tiny_m_into,
        linear_w8a16_tiny_m_into, quantize_weight_s8_per_channel,
    },
    tensor::Shape,
    weights::WeightStore,
};

const BF16_DOWN_CANDIDATE_LAYERS: [usize; 9] = [0, 1, 2, 3, 4, 5, 7, 11, 13];

fn model_dir() -> PathBuf {
    env::var_os("LFM25_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/LFM2.5-1.2B-Instruct"))
}

fn cosine_similarity(actual: &[bf16], reference: &[bf16]) -> f64 {
    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut reference_norm = 0.0f64;
    for (&actual, &reference) in actual.iter().zip(reference) {
        let actual = f64::from(actual.to_f32());
        let reference = f64::from(reference.to_f32());
        dot += actual * reference;
        actual_norm += actual * actual;
        reference_norm += reference * reference;
    }
    dot / (actual_norm * reference_norm)
        .sqrt()
        .max(f64::MIN_POSITIVE)
}

fn relative_l2(actual: &[bf16], reference: &[bf16]) -> f64 {
    let mut error = 0.0f64;
    let mut reference_norm = 0.0f64;
    for (&actual, &reference) in actual.iter().zip(reference) {
        let actual = f64::from(actual.to_f32());
        let reference = f64::from(reference.to_f32());
        let delta = actual - reference;
        error += delta * delta;
        reference_norm += reference * reference;
    }
    (error / reference_norm.max(f64::MIN_POSITIVE)).sqrt()
}

fn collect_real_decode_samples(
    runtime: &CudaRuntime,
    model: &Lfm2Model,
) -> Result<CalibrationCollector> {
    const PREFILL_TOKENS: usize = 64;
    const DECODE_SAMPLES: usize = 8;
    let mut cache = model.new_cache(
        runtime,
        PREFILL_TOKENS + DECODE_SAMPLES + 4,
        KvPageSize::P16,
    )?;
    let mut prompt = Vec::with_capacity(PREFILL_TOKENS);
    prompt.push(model.config().bos_token_id);
    prompt.extend((1..PREFILL_TOKENS).map(|index| 100u32 + u32::try_from(index).unwrap()));
    let _ = model.forward_logits(runtime, &mut cache, &prompt)?;

    let mut calibration = CalibrationCollector::new(DECODE_SAMPLES);
    calibration.set_activation_phase(CalibrationPhase::Decode)?;
    for step in 0..DECODE_SAMPLES {
        let token = 1000u32 + u32::try_from(step)?;
        let _ = model.forward_logits_calibrated(runtime, &mut cache, &[token], &mut calibration)?;
    }
    runtime.synchronize()?;
    Ok(calibration)
}

#[test]
#[ignore = "real checkpoint W8A16 GPU benchmark"]
fn bench_w8a16_tiny_m_real_down_sites() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let model_dir = model_dir();

    let mut store = WeightStore::load(&runtime, &model_dir)?;
    let mut down_weights = Vec::with_capacity(BF16_DOWN_CANDIDATE_LAYERS.len());
    for layer in BF16_DOWN_CANDIDATE_LAYERS {
        down_weights.push((
            layer,
            store.take(&format!("model.layers.{layer}.feed_forward.w2.weight"))?,
        ));
    }
    drop(store);

    let model = Lfm2Model::load(&runtime, &model_dir)?;
    let calibration = collect_real_decode_samples(&runtime, &model)?;
    let bench = BenchConfig {
        warmup: 8,
        batches: 24,
        iterations_per_batch: 10,
    };

    for (layer, weight) in &down_weights {
        let sample_name = format!("layers.{layer}.mlp.down.input");
        let (samples, rows, k) = calibration
            .decode_samples(&sample_name)
            .with_context(|| format!("missing decode samples for {sample_name}"))?;
        ensure!(rows >= 2, "need at least two real decode rows for {sample_name}");
        ensure!(
            weight.rank() == 2 && weight.dims()[1] == k,
            "real W8A16 down weight shape mismatch"
        );
        let n = weight.dims()[0];
        let quantized_weight = quantize_weight_s8_per_channel(&runtime, weight)?;

        for m in [1usize, 2] {
            runtime.blaslt().prepare_linear_bf16(m, n, k)?;
            let input = runtime.upload(&samples[..m * k], Shape::new([m, k]))?;
            let mut bf16_output = runtime.alloc_bf16(Shape::new([m, n]))?;
            let mut w8a8_output = runtime.alloc_bf16(Shape::new([m, n]))?;
            let mut w8a16_output = runtime.alloc_bf16(Shape::new([m, n]))?;
            let mut w8a8_workspace = Int8TinyMWorkspace::new(&runtime, 2, k)?;

            linear_bf16_into(&runtime, &input, weight, &mut bf16_output)?;
            linear_int8_tiny_m_into(
                &runtime,
                &input,
                &quantized_weight,
                &mut w8a8_workspace,
                &mut w8a8_output,
            )?;
            linear_w8a16_tiny_m_into(
                &runtime,
                &input,
                &quantized_weight,
                &mut w8a16_output,
            )?;
            runtime.synchronize()?;

            let reference = readback(&runtime, &bf16_output)?;
            let w8a8 = readback(&runtime, &w8a8_output)?;
            let w8a16 = readback(&runtime, &w8a16_output)?;
            let w8a8_cosine = cosine_similarity(&w8a8, &reference);
            let w8a8_rel_l2 = relative_l2(&w8a8, &reference);
            let w8a16_cosine = cosine_similarity(&w8a16, &reference);
            let w8a16_rel_l2 = relative_l2(&w8a16, &reference);

            let w8a16_vs_bf16 = benchmark_gpu_paired(
                runtime.context(),
                runtime.stream(),
                bench,
                || linear_bf16_into(&runtime, &input, weight, &mut bf16_output),
                || {
                    linear_w8a16_tiny_m_into(
                        &runtime,
                        &input,
                        &quantized_weight,
                        &mut w8a16_output,
                    )
                },
            )?;

            println!(
                "w8a16_real_down layer={} m={} w8a8_cosine={:.6} w8a8_rel_l2={:.6} w8a16_cosine={:.6} w8a16_rel_l2={:.6} bf16_mean_us={:.3} w8a16_mean_us={:.3} w8a16_speedup={:.4}x w8a16_p95_us={:.3}",
                layer,
                m,
                w8a8_cosine,
                w8a8_rel_l2,
                w8a16_cosine,
                w8a16_rel_l2,
                w8a16_vs_bf16.reference.mean_us,
                w8a16_vs_bf16.candidate.mean_us,
                w8a16_vs_bf16.speedup_mean,
                w8a16_vs_bf16.candidate.p95_us,
            );
        }
    }

    Ok(())
}
