use std::{env, fs, path::PathBuf};

use anyhow::{Context as _, Result, ensure};
use half::bf16;

use crate::{
    cache::KvPageSize,
    cuda::{
        CudaRuntime,
        benchmark::{BenchConfig, benchmark_gpu_paired},
    },
    model::{Fp8PrecisionPolicy, HiddenCapture, PropagationAccumulator},
    ops::prefill_dispatch::ScopedFlashPrefillOverride,
};

use super::*;

const PROMPT_SHAPES: &[usize] = &[516, 2056, 8202];

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

fn generate_prompt_tokens(model: &Lfm2Model, context: usize) -> Result<Vec<u32>> {
    let mut tokens = Vec::with_capacity(context);
    for position in 0..context {
        let token = if position == 0 {
            model.config().bos_token_id
        } else {
            100u32 + u32::try_from((position * 13 + 37) % 4000)?
        };
        tokens.push(token);
    }
    Ok(tokens)
}

struct LogitMetrics {
    cosine: f64,
    nrmse: f64,
    top1_match: bool,
    non_finite: usize,
}

fn compare_logits(reference: &[bf16], candidate: &[bf16]) -> LogitMetrics {
    assert_eq!(reference.len(), candidate.len());
    let mut dot = 0.0f64;
    let mut ref_sq = 0.0f64;
    let mut cand_sq = 0.0f64;
    let mut err_sq = 0.0f64;
    let mut non_finite = 0usize;
    let mut ref_max_val = f32::NEG_INFINITY;
    let mut ref_max_idx = 0usize;
    let mut cand_max_val = f32::NEG_INFINITY;
    let mut cand_max_idx = 0usize;

    for (idx, (&r_bf, &c_bf)) in reference.iter().zip(candidate).enumerate() {
        let r = f64::from(r_bf.to_f32());
        let c = f64::from(c_bf.to_f32());
        if !r.is_finite() || !c.is_finite() {
            non_finite += 1;
            continue;
        }
        dot += r * c;
        ref_sq += r * r;
        cand_sq += c * c;
        let diff = c - r;
        err_sq += diff * diff;

        let r_f32 = r_bf.to_f32();
        if r_f32 > ref_max_val {
            ref_max_val = r_f32;
            ref_max_idx = idx;
        }
        let c_f32 = c_bf.to_f32();
        if c_f32 > cand_max_val {
            cand_max_val = c_f32;
            cand_max_idx = idx;
        }
    }

    let cosine = if ref_sq > 0.0 && cand_sq > 0.0 {
        dot / (ref_sq * cand_sq).sqrt()
    } else {
        0.0
    };
    let nrmse = if ref_sq > 0.0 {
        (err_sq / ref_sq).sqrt()
    } else {
        0.0
    };
    let top1_match = ref_max_idx == cand_max_idx;

    LogitMetrics {
        cosine,
        nrmse,
        top1_match,
        non_finite,
    }
}

#[test]
#[ignore = "GPU benchmark: real-checkpoint full-model FlashAttention prefill ABBA benchmark"]
fn bench_prefill_flash_production_abba() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let model = load_model(&runtime)?;
    let page_size = KvPageSize::P16;

    for &context in PROMPT_SHAPES {
        let token_ids = generate_prompt_tokens(&model, context)?;
        let capacity = context + 64;

        // Baseline run (Q2) with hidden capture
        let mut ref_cache = model.new_cache(&runtime, capacity, page_size)?;
        let mut ref_capture = HiddenCapture::default();
        let ref_logits_tensor = {
            let _guard = ScopedFlashPrefillOverride::new(false);
            model.forward_logits_captured(&runtime, &mut ref_cache, &token_ids, &mut ref_capture)?
        };
        runtime.synchronize()?;
        let ref_logits = runtime.download(&ref_logits_tensor)?;

        // Candidate run (FlashAttention) with hidden capture
        let mut cand_cache = model.new_cache(&runtime, capacity, page_size)?;
        let mut cand_capture = HiddenCapture::default();
        let cand_logits_tensor = {
            let _guard = ScopedFlashPrefillOverride::new(true);
            model.forward_logits_captured(
                &runtime,
                &mut cand_cache,
                &token_ids,
                &mut cand_capture,
            )?
        };
        runtime.synchronize()?;
        let cand_logits = runtime.download(&cand_logits_tensor)?;

        // Phase 1 Hidden Layer Quality Gate
        let mut prop_accum = PropagationAccumulator::default();
        prop_accum.observe(&ref_capture, &cand_capture)?;
        let layer_metrics = prop_accum.finish();
        let mut min_hidden_cos = 1.0f64;
        let mut max_hidden_nrmse = 0.0f64;
        for point in &layer_metrics {
            ensure!(
                point.non_finite_values == 0,
                "hidden point {} has non-finite values at N={context}",
                point.point
            );
            ensure!(
                point.cosine >= 0.99,
                "hidden point {} cosine {:.6} < 0.99 at N={context}",
                point.point,
                point.cosine
            );
            ensure!(
                point.nrmse <= 0.10,
                "hidden point {} NRMSE {:.6} > 0.10 at N={context}",
                point.point,
                point.nrmse
            );
            if point.cosine < min_hidden_cos {
                min_hidden_cos = point.cosine;
            }
            if point.nrmse > max_hidden_nrmse {
                max_hidden_nrmse = point.nrmse;
            }
        }

        // Logit fidelity
        let metrics = compare_logits(&ref_logits, &cand_logits);
        ensure!(
            metrics.non_finite == 0,
            "full model produced non-finite logits at N={context}"
        );
        ensure!(
            metrics.cosine >= 0.99,
            "full model cosine {cos:.6} < 0.99 gate at N={context}",
            cos = metrics.cosine
        );
        ensure!(
            metrics.nrmse <= 0.10,
            "full model NRMSE {nrmse:.6} > 0.10 gate at N={context}",
            nrmse = metrics.nrmse
        );

        // Paired ABBA benchmark timing
        let mut bench_ref_cache = model.new_cache(&runtime, capacity, page_size)?;
        let mut bench_cand_cache = model.new_cache(&runtime, capacity, page_size)?;

        let stats = benchmark_gpu_paired(
            runtime.context(),
            runtime.stream(),
            BenchConfig {
                warmup: 2,
                batches: 10,
                iterations_per_batch: 1,
            },
            || {
                bench_ref_cache.reset(&runtime)?;
                let _guard = ScopedFlashPrefillOverride::new(false);
                let _ = model.forward_logits(&runtime, &mut bench_ref_cache, &token_ids)?;
                Ok(())
            },
            || {
                bench_cand_cache.reset(&runtime)?;
                let _guard = ScopedFlashPrefillOverride::new(true);
                let _ = model.forward_logits(&runtime, &mut bench_cand_cache, &token_ids)?;
                Ok(())
            },
        )?;

        println!(
            "prefill_flash_e2e N={} q2_mean_ms={:.3} flash_mean_ms={:.3} speedup_mean={:.4}x q2_p50_ms={:.3} flash_p50_ms={:.3} speedup_p50={:.4}x q2_p95_ms={:.3} flash_p95_ms={:.3} speedup_p95={:.4}x logit_cosine={:.6} logit_nrmse={:.6} min_hidden_cos={:.6} max_hidden_nrmse={:.6} top1_match={}",
            context,
            stats.reference.mean_us / 1000.0,
            stats.candidate.mean_us / 1000.0,
            stats.speedup_mean,
            stats.reference.p50_us / 1000.0,
            stats.candidate.p50_us / 1000.0,
            stats.speedup_p50,
            stats.reference.p95_us / 1000.0,
            stats.candidate.p95_us / 1000.0,
            stats.speedup_p95,
            metrics.cosine,
            metrics.nrmse,
            min_hidden_cos,
            max_hidden_nrmse,
            metrics.top1_match,
        );
    }

    Ok(())
}
