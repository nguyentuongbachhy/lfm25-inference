use std::{env, fs, path::PathBuf};

use anyhow::{Context as _, Result, ensure};
use half::bf16;

use crate::{
    cache::KvPageSize,
    cuda::{
        CudaRuntime,
        benchmark::{BenchConfig, benchmark_gpu_paired},
    },
    model::{Fp8PrecisionPolicy, RaggedBatchInput},
    ops::prefill_dispatch::ScopedFlashPrefillOverride,
};

use super::*;

struct RaggedShape {
    batch_size: usize,
    tokens_per_seq: usize,
}

const RAGGED_SHAPES: &[RaggedShape] = &[
    RaggedShape {
        batch_size: 2,
        tokens_per_seq: 512,
    },
    RaggedShape {
        batch_size: 4,
        tokens_per_seq: 512,
    },
    RaggedShape {
        batch_size: 2,
        tokens_per_seq: 2048,
    },
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
    let enabled = model.install_fp8_policy(runtime, &policy)?;
    ensure!(enabled > 0, "selected FP8 policy enables no sites");
    Ok(model)
}

type RaggedInputTuple = (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>);

fn build_ragged_input(
    model: &Lfm2Model,
    batch_size: usize,
    tokens_per_seq: usize,
) -> Result<RaggedInputTuple> {
    let total_tokens = batch_size * tokens_per_seq;
    let mut token_ids = Vec::with_capacity(total_tokens);
    let mut positions = Vec::with_capacity(total_tokens);
    let mut request_slots = Vec::with_capacity(total_tokens);
    let mut segment_offsets = Vec::with_capacity(batch_size + 1);
    let mut segment_slots = Vec::with_capacity(batch_size);
    let mut output_rows = Vec::with_capacity(batch_size);

    segment_offsets.push(0u32);
    for b in 0..batch_size {
        for pos in 0..tokens_per_seq {
            let token = if pos == 0 {
                model.config().bos_token_id
            } else {
                100u32 + u32::try_from(((b * 1000 + pos) * 17 + 43) % 5000)?
            };
            token_ids.push(token);
            positions.push(u32::try_from(pos)?);
            request_slots.push(u32::try_from(b)?);
        }
        segment_offsets.push(u32::try_from(token_ids.len())?);
        segment_slots.push(u32::try_from(b)?);
        output_rows.push(u32::try_from(token_ids.len() - 1)?);
    }

    Ok((
        token_ids,
        positions,
        request_slots,
        segment_offsets,
        segment_slots,
        output_rows,
    ))
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
#[ignore = "GPU benchmark: real-checkpoint full-model Segmented FlashAttention prefill ABBA benchmark"]
fn bench_ragged_flash_production_abba() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let model = load_model(&runtime)?;
    let page_size = KvPageSize::P16;

    for shape in RAGGED_SHAPES {
        let batch_size = shape.batch_size;
        let tokens_per_seq = shape.tokens_per_seq;
        let total_tokens = batch_size * tokens_per_seq;

        let (token_ids, positions, request_slots, segment_offsets, segment_slots, output_rows) =
            build_ragged_input(&model, batch_size, tokens_per_seq)?;

        let pages = batch_size * tokens_per_seq.div_ceil(page_size.value());

        // Baseline (Flash disabled -> hybrid ragged scalar loop)
        let mut ref_cache =
            model.new_batch_cache(&runtime, batch_size, total_tokens, pages, page_size)?;
        for slot in 0..batch_size {
            ref_cache.reserve(slot, tokens_per_seq)?;
        }
        let ref_logits_tensor = {
            let _guard = ScopedFlashPrefillOverride::new(false);
            model.forward_ragged_batch(
                &runtime,
                &mut ref_cache,
                RaggedBatchInput {
                    token_ids: &token_ids,
                    positions: &positions,
                    request_slots: &request_slots,
                    segment_offsets: &segment_offsets,
                    segment_slots: &segment_slots,
                    output_rows: &output_rows,
                },
            )?
        };
        runtime.synchronize()?;
        let ref_logits = runtime.download(&ref_logits_tensor)?;

        // Candidate (Segmented FlashAttention enabled)
        let mut cand_cache =
            model.new_batch_cache(&runtime, batch_size, total_tokens, pages, page_size)?;
        for slot in 0..batch_size {
            cand_cache.reserve(slot, tokens_per_seq)?;
        }
        let cand_logits_tensor = {
            let _guard = ScopedFlashPrefillOverride::new(true);
            model.forward_ragged_batch(
                &runtime,
                &mut cand_cache,
                RaggedBatchInput {
                    token_ids: &token_ids,
                    positions: &positions,
                    request_slots: &request_slots,
                    segment_offsets: &segment_offsets,
                    segment_slots: &segment_slots,
                    output_rows: &output_rows,
                },
            )?
        };
        runtime.synchronize()?;
        let cand_logits = runtime.download(&cand_logits_tensor)?;

        // Check numerical quality per sequence
        let vocab_size = model.config().vocab_size;
        for b in 0..batch_size {
            let ref_seq = &ref_logits[b * vocab_size..(b + 1) * vocab_size];
            let cand_seq = &cand_logits[b * vocab_size..(b + 1) * vocab_size];
            let metrics = compare_logits(ref_seq, cand_seq);

            println!(
                "[RAGGED FLASH ABBA] B={batch_size} L={tokens_per_seq} seq={b}: cosine={:.6}, nrmse={:.6}, top1={}, non_finite={}",
                metrics.cosine, metrics.nrmse, metrics.top1_match, metrics.non_finite
            );

            ensure!(
                metrics.non_finite == 0,
                "B={batch_size} L={tokens_per_seq} seq={b} produced {} non-finite logits",
                metrics.non_finite
            );
            ensure!(
                metrics.cosine >= 0.999,
                "B={batch_size} L={tokens_per_seq} seq={b} cosine {:.6} < 0.999",
                metrics.cosine
            );
            ensure!(
                metrics.nrmse <= 0.05,
                "B={batch_size} L={tokens_per_seq} seq={b} NRMSE {:.6} > 0.05",
                metrics.nrmse
            );
            ensure!(
                metrics.top1_match,
                "B={batch_size} L={tokens_per_seq} seq={b} top-1 argmax prediction mismatch"
            );
        }

        // Run Paired ABBA Benchmark
        let bench_cfg = BenchConfig {
            warmup: 2,
            batches: 10,
            iterations_per_batch: 1,
        };

        let mut bench_ref_cache =
            model.new_batch_cache(&runtime, batch_size, total_tokens, pages, page_size)?;
        let mut bench_cand_cache =
            model.new_batch_cache(&runtime, batch_size, total_tokens, pages, page_size)?;

        let stats = benchmark_gpu_paired(
            runtime.context(),
            runtime.stream(),
            bench_cfg,
            // A: Legacy hybrid ragged attention
            || {
                for slot in 0..batch_size {
                    bench_ref_cache.release(&runtime, slot)?;
                    bench_ref_cache.reserve(slot, tokens_per_seq)?;
                }
                let _guard = ScopedFlashPrefillOverride::new(false);
                let _ = model.forward_ragged_batch(
                    &runtime,
                    &mut bench_ref_cache,
                    RaggedBatchInput {
                        token_ids: &token_ids,
                        positions: &positions,
                        request_slots: &request_slots,
                        segment_offsets: &segment_offsets,
                        segment_slots: &segment_slots,
                        output_rows: &output_rows,
                    },
                )?;
                Ok(())
            },
            // B: Segmented Tensor Core FlashAttention
            || {
                for slot in 0..batch_size {
                    bench_cand_cache.release(&runtime, slot)?;
                    bench_cand_cache.reserve(slot, tokens_per_seq)?;
                }
                let _guard = ScopedFlashPrefillOverride::new(true);
                let _ = model.forward_ragged_batch(
                    &runtime,
                    &mut bench_cand_cache,
                    RaggedBatchInput {
                        token_ids: &token_ids,
                        positions: &positions,
                        request_slots: &request_slots,
                        segment_offsets: &segment_offsets,
                        segment_slots: &segment_slots,
                        output_rows: &output_rows,
                    },
                )?;
                Ok(())
            },
        )?;

        println!(
            "ragged_flash_e2e B={} L={} N={} legacy_mean_ms={:.3} flash_mean_ms={:.3} speedup_mean={:.4}x legacy_p50_ms={:.3} flash_p50_ms={:.3} speedup_p50={:.4}x legacy_p95_ms={:.3} flash_p95_ms={:.3} speedup_p95={:.4}x",
            batch_size,
            tokens_per_seq,
            total_tokens,
            stats.reference.mean_us / 1000.0,
            stats.candidate.mean_us / 1000.0,
            stats.speedup_mean,
            stats.reference.p50_us / 1000.0,
            stats.candidate.p50_us / 1000.0,
            stats.speedup_p50,
            stats.reference.p95_us / 1000.0,
            stats.candidate.p95_us / 1000.0,
            stats.speedup_p95,
        );

        ensure!(
            stats.speedup_mean >= 1.05,
            "B={batch_size} L={tokens_per_seq} failed speedup gate: {:.4}x < 1.05x",
            stats.speedup_mean
        );
    }

    Ok(())
}
