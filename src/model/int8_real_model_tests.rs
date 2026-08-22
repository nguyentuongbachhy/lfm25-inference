use std::{env, fs, path::PathBuf};

use anyhow::{Context as _, Result, ensure};
use half::bf16;

use crate::{
    cache::KvPageSize,
    config::Lfm2Config,
    cuda::{
        CudaRuntime, Fp8ScaleMode,
        benchmark::{BenchConfig, benchmark_gpu_paired},
        testing::readback,
    },
    model::{CalibrationCollector, CalibrationPhase, Fp8PrecisionPolicy, Lfm2Model},
    ops::{
        Int8TinyMWorkspace, linear_bf16_into, linear_fp8_e4m3,
        linear_int8_tiny_m_into, quantize_weight_e4m3, quantize_weight_s8_per_channel,
    },
    tensor::Shape,
    weights::WeightStore,
};

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
#[ignore = "real checkpoint GPU benchmark"]
fn bench_int8_tiny_m_real_down_sites() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let model_dir = model_dir();
    let config = Lfm2Config::from_model_dir(&model_dir)?;

    // Keep only the real w2/down matrices so the temporary WeightStore can be
    // released before loading the model used to capture real decode activations.
    let mut store = WeightStore::load(&runtime, &model_dir)?;
    let mut down_weights = Vec::with_capacity(config.num_hidden_layers);
    for layer in 0..config.num_hidden_layers {
        down_weights.push(store.take(&format!(
            "model.layers.{layer}.feed_forward.w2.weight"
        ))?);
    }
    drop(store);

    let model = Lfm2Model::load(&runtime, &model_dir)?;
    let calibration = collect_real_decode_samples(&runtime, &model)?;
    let policy_bytes = fs::read(policy_path()).context("failed to read selected FP8 policy")?;
    let policy: Fp8PrecisionPolicy =
        serde_json::from_slice(&policy_bytes).context("failed to parse selected FP8 policy")?;

    let bench = BenchConfig {
        warmup: 8,
        batches: 24,
        iterations_per_batch: 10,
    };

    for (layer, weight) in down_weights.iter().enumerate() {
        let sample_name = format!("layers.{layer}.mlp.down.input");
        let (samples, rows, k) = calibration
            .decode_samples(&sample_name)
            .with_context(|| format!("missing decode samples for {sample_name}"))?;
        ensure!(rows >= 2, "need at least two real decode rows for {sample_name}");
        ensure!(weight.rank() == 2 && weight.dims()[1] == k, "real down weight shape mismatch");
        let n = weight.dims()[0];
        let int8_weight = quantize_weight_s8_per_channel(&runtime, weight)?;
        let site_name = format!("layers.{layer}.mlp.down");
        let fp8_site = policy.sites.iter().find(|site| site.site == site_name);

        for m in [1usize, 2] {
            runtime.blaslt().prepare_linear_bf16(m, n, k)?;
            let input = runtime.upload(&samples[..m * k], Shape::new([m, k]))?;
            let mut bf16_output = runtime.alloc_bf16(Shape::new([m, n]))?;
            let mut int8_output = runtime.alloc_bf16(Shape::new([m, n]))?;
            let mut int8_workspace = Int8TinyMWorkspace::new(&runtime, 2, k)?;

            linear_bf16_into(&runtime, &input, weight, &mut bf16_output)?;
            linear_int8_tiny_m_into(
                &runtime,
                &input,
                &int8_weight,
                &mut int8_workspace,
                &mut int8_output,
            )?;
            runtime.synchronize()?;
            let bf16_host = readback(&runtime, &bf16_output)?;
            let int8_host = readback(&runtime, &int8_output)?;
            let int8_cosine = cosine_similarity(&int8_host, &bf16_host);
            let int8_rel_l2 = relative_l2(&int8_host, &bf16_host);

            let int8_vs_bf16 = benchmark_gpu_paired(
                runtime.context(),
                runtime.stream(),
                bench,
                || linear_bf16_into(&runtime, &input, weight, &mut bf16_output),
                || {
                    linear_int8_tiny_m_into(
                        &runtime,
                        &input,
                        &int8_weight,
                        &mut int8_workspace,
                        &mut int8_output,
                    )
                },
            )?;

            if let Some(site) = fp8_site.filter(|site| site.enabled) {
                runtime
                    .blaslt()
                    .prepare_linear_fp8(m, n, k, Fp8ScaleMode::Tensorwide)?;
                let fp8_weight = quantize_weight_e4m3(&runtime, weight, site.weight_scale)?;
                let fp8_output = linear_fp8_e4m3(
                    &runtime,
                    &input,
                    &fp8_weight,
                    site.activation_scale,
                    site.weight_scale,
                )?;
                runtime.synchronize()?;
                let fp8_host = readback(&runtime, &fp8_output)?;
                let fp8_cosine = cosine_similarity(&fp8_host, &bf16_host);
                let fp8_rel_l2 = relative_l2(&fp8_host, &bf16_host);

                let int8_vs_fp8_generic = benchmark_gpu_paired(
                    runtime.context(),
                    runtime.stream(),
                    bench,
                    || {
                        let output = linear_fp8_e4m3(
                            &runtime,
                            &input,
                            &fp8_weight,
                            site.activation_scale,
                            site.weight_scale,
                        )?;
                        drop(output);
                        Ok(())
                    },
                    || {
                        linear_int8_tiny_m_into(
                            &runtime,
                            &input,
                            &int8_weight,
                            &mut int8_workspace,
                            &mut int8_output,
                        )
                    },
                )?;

                println!(
                    "int8_real_down layer={} m={} fp8_enabled=true int8_cosine={:.6} int8_rel_l2={:.6} fp8_cosine={:.6} fp8_rel_l2={:.6} bf16_mean_us={:.3} int8_e2e_mean_us={:.3} int8_vs_bf16={:.4}x fp8_generic_mean_us={:.3} int8_vs_fp8_generic={:.4}x",
                    layer,
                    m,
                    int8_cosine,
                    int8_rel_l2,
                    fp8_cosine,
                    fp8_rel_l2,
                    int8_vs_bf16.reference.mean_us,
                    int8_vs_bf16.candidate.mean_us,
                    int8_vs_bf16.speedup_mean,
                    int8_vs_fp8_generic.reference.mean_us,
                    int8_vs_fp8_generic.speedup_mean,
                );
            } else {
                println!(
                    "int8_real_down layer={} m={} fp8_enabled=false int8_cosine={:.6} int8_rel_l2={:.6} bf16_mean_us={:.3} int8_e2e_mean_us={:.3} int8_vs_bf16={:.4}x",
                    layer,
                    m,
                    int8_cosine,
                    int8_rel_l2,
                    int8_vs_bf16.reference.mean_us,
                    int8_vs_bf16.candidate.mean_us,
                    int8_vs_bf16.speedup_mean,
                );
            }
        }
    }

    Ok(())
}
