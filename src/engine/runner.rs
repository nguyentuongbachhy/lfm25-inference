use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context as _, Result, ensure};
use serde::Serialize;

use crate::{
    cache::KvPageSize,
    cuda::CudaRuntime,
    generation::{DEFAULT_SAMPLING_SEED, Sampler, SamplingConfig},
    model::{
        CalibrationCollector, CalibrationPhase, DecodeProfileMode, DecodeProfileReport,
        Fp8CalibrationReport, Fp8GemmErrorReport, Fp8PrecisionPolicy, HiddenCapture, Lfm2Model,
        LogitDistributionMetrics, LogitMetricAccumulator, ModelProfileRecorder, PrecisionClass,
        ProfileRegion, PropagationAccumulator, PropagationPointMetrics, RaggedBatchInput,
    },
    scheduler::{CostCurve, CostPoint, HardwareCostModel},
    tokenizer::Lfm2Tokenizer,
};

#[derive(Debug, Clone, Copy)]
pub struct GenerationOptions {
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 64,
            sampling: SamplingConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EngineConfig {
    pub kv_page_size: KvPageSize,
    pub decode_profile: DecodeProfileMode,
    pub decode_profile_warmup_steps: usize,
    pub decode_profile_steps: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            kv_page_size: KvPageSize::P16,
            decode_profile: DecodeProfileMode::Off,
            decode_profile_warmup_steps: 4,
            decode_profile_steps: 128,
        }
    }
}

#[derive(Debug)]
pub struct GenerationResult {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: &'static str,
    pub metrics: GenerationMetrics,
    pub profile: Option<DecodeProfileReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GenerationMetrics {
    pub tokenization_ms: f64,
    pub queue_delay_ms: f64,
    pub scheduler_cpu_ms: f64,
    pub cache_allocation_cpu_ms: f64,
    pub cache_initialization_gpu_ms: f64,
    pub prefill_gpu_ms: f64,
    pub prefill_submit_cpu_ms: f64,
    pub prefill_d2h_ms: f64,
    pub first_token_gpu_wait_and_sampling_ms: f64,
    pub ttft_ms: f64,
    pub decode_gpu_ms: f64,
    pub decode_submit_cpu_ms: f64,
    pub decode_d2h_ms: f64,
    pub decode_total_ms: f64,
    pub tpot_mean_ms: Option<f64>,
    pub tpot_p50_ms: Option<f64>,
    pub tpot_p95_ms: Option<f64>,
    pub decode_tokens_per_second: Option<f64>,
    pub gpu_wait_and_sampling_total_ms: f64,
    pub bf16_pool_hits: u64,
    pub bf16_pool_misses: u64,
    pub fp8_pool_hits: u64,
    pub fp8_pool_misses: u64,
    pub decode_bf16_pool_hits: u64,
    pub decode_bf16_pool_misses: u64,
    pub decode_fp8_pool_hits: u64,
    pub decode_fp8_pool_misses: u64,
    pub bf16_pool_available_elements: usize,
    pub bf16_pool_dropped_elements: u64,
    pub fp8_pool_available_elements: usize,
    pub fp8_pool_dropped_elements: u64,
    pub bf16_pool_internal_fragment_elements: u64,
    pub fp8_pool_internal_fragment_elements: u64,
    pub detokenization_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Fp8BenchmarkSummary {
    pub ttft_mean_ms: f64,
    pub ttft_p50_ms: f64,
    pub ttft_p95_ms: f64,
    pub tpot_mean_ms: f64,
    pub tpot_p50_ms: f64,
    pub tpot_p95_ms: f64,
    pub total_mean_ms: f64,
    pub bf16_pool_hits: u64,
    pub bf16_pool_misses: u64,
    pub fp8_pool_hits: u64,
    pub fp8_pool_misses: u64,
    pub decode_bf16_pool_hits: u64,
    pub decode_bf16_pool_misses: u64,
    pub decode_fp8_pool_hits: u64,
    pub decode_fp8_pool_misses: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Fp8BenchmarkWorkload {
    pub requested_context_tokens: usize,
    pub actual_prompt_tokens: usize,
    pub completion_tokens: usize,
    pub warmup_pairs: usize,
    pub measured_pairs: usize,
    pub bf16: Fp8BenchmarkSummary,
    pub fp8: Fp8BenchmarkSummary,
    pub paired_tpot_speedup_mean: f64,
    pub paired_tpot_speedup_p50: f64,
    pub paired_tpot_speedup_p95: f64,
    pub paired_tpot_speedup_min: f64,
    pub paired_tpot_speedup_max: f64,
    pub paired_ttft_ratio_mean: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Fp8BenchmarkReport {
    pub schema_version: u32,
    pub design: &'static str,
    pub warmup_pairs: usize,
    pub measured_pairs: usize,
    pub completion_tokens: usize,
    pub workloads: Vec<Fp8BenchmarkWorkload>,
}

#[derive(Debug, Serialize)]
pub struct ServingDecodePoint {
    pub batch_size: usize,
    pub context_tokens: usize,
    pub precision: &'static str,
    pub warmup_steps: usize,
    pub measured_steps: usize,
    pub step_mean_ms: f64,
    pub step_p50_ms: f64,
    pub step_p95_ms: f64,
    pub output_tokens_per_second: f64,
    pub goodput_tokens_per_second: f64,
    pub tpot_slo_pass: bool,
    pub kv_pages_allocated: usize,
    pub kv_internal_fragmentation_ratio: f64,
    pub h2d_bytes_per_output_token: f64,
    pub h2d_calls_per_step: f64,
    pub bf16_pool_misses_after_warmup: u64,
    pub fp8_pool_misses_after_warmup: u64,
    pub identical_sequence_row_nrmse_max: f64,
    pub identical_sequence_top1_agreement: bool,
}

#[derive(Debug, Serialize)]
pub struct ServingDecodeBenchmarkReport {
    pub schema_version: u32,
    pub design: &'static str,
    pub page_size: usize,
    pub tpot_slo_ms: f64,
    pub ragged_prefill_validation: RaggedPrefillValidation,
    pub points: Vec<ServingDecodePoint>,
    pub skipped_capacity_points: Vec<ServingSkippedPoint>,
}

#[derive(Debug, Serialize)]
pub struct ServingSkippedPoint {
    pub batch_size: usize,
    pub context_tokens: usize,
    pub required_kv_bytes: usize,
    pub free_vram_bytes: usize,
    pub reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct RaggedPrefillValidation {
    pub tokens_per_sequence: usize,
    pub batch_size: usize,
    pub legacy_contiguous_ms: f64,
    pub ragged_paged_ms: f64,
    pub final_logits_nrmse_max: f64,
    pub final_logits_cosine_min: f64,
    pub top1_agreement: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ServingPrefillPoint {
    pub prompt_tokens: usize,
    pub warmup_steps: usize,
    pub measured_steps: usize,
    pub wall_mean_ms: f64,
    pub wall_p50_ms: f64,
    pub wall_p95_ms: f64,
    pub gpu_mean_ms: f64,
    pub gpu_p95_ms: f64,
    pub submit_cpu_mean_ms: f64,
    pub bf16_pool_misses_after_warmup: u64,
    pub bf16_pool_dropped_elements: u64,
}

#[derive(Debug, Serialize)]
pub struct HardwareProfileBenchmarkReport {
    pub schema_version: u32,
    pub gpu_name: String,
    pub page_size: usize,
    pub free_vram_bytes_before_benchmark: usize,
    pub total_vram_bytes: usize,
    pub interactive_ttft_headroom_ms: f64,
    pub decode: ServingDecodeBenchmarkReport,
    pub prefill: Vec<ServingPrefillPoint>,
    pub cost_model: HardwareCostModel,
}

#[derive(Debug, Serialize)]
pub struct BatchedFp8Point {
    pub batch_size: usize,
    pub context_tokens: usize,
    pub warmup_pairs: usize,
    pub measured_pairs: usize,
    pub bf16_mean_ms: f64,
    pub bf16_p95_ms: f64,
    pub fp8_mean_ms: f64,
    pub fp8_p95_ms: f64,
    pub paired_speedup_mean: f64,
    pub logit_nrmse: f64,
    pub logit_cosine: f64,
    pub top1_agreement_ratio: f64,
    pub non_finite_logits: usize,
    pub smoke_quality_pass: bool,
    pub performance_pass: bool,
}

#[derive(Debug, Serialize)]
pub struct BatchedFp8BenchmarkReport {
    pub schema_version: u32,
    pub format: &'static str,
    pub history_mode: &'static str,
    pub quality_scope: &'static str,
    pub promotion_gate: &'static str,
    pub points: Vec<BatchedFp8Point>,
    pub all_smoke_quality_pass: bool,
    pub all_performance_pass: bool,
    pub promoted: bool,
}

pub struct Engine {
    pub(super) runtime: CudaRuntime,
    pub(super) model: Lfm2Model,
    pub(super) tokenizer: Lfm2Tokenizer,
    pub(super) config: EngineConfig,
    model_dir: PathBuf,
}

pub struct Fp8CalibrationArtifacts {
    pub calibration: Fp8CalibrationReport,
    pub gemm_error: Fp8GemmErrorReport,
    pub policies: Vec<Fp8PrecisionPolicy>,
    pub quality: Fp8QualityStudyReport,
    pub sensitivity: Fp8SensitivityReport,
    pub policy_search: Fp8PolicySearchReport,
    pub selected_policy: Option<Fp8PrecisionPolicy>,
}

#[derive(Debug, Serialize)]
pub struct Fp8FinalValidationReport {
    pub schema_version: u32,
    pub corpus_path: String,
    pub history_mode: &'static str,
    pub quality: Fp8PolicyQualityReport,
    pub greedy: Fp8GreedyDiagnostics,
}

#[derive(Debug, Serialize)]
pub struct Fp8PositionQuality {
    pub input_position: usize,
    pub metrics: LogitDistributionMetrics,
}

#[derive(Debug, Serialize)]
pub struct Fp8PolicyQualityReport {
    pub policy_name: String,
    pub enabled_sites: usize,
    pub evaluation_sequences: usize,
    pub evaluation_source_tokens: usize,
    pub metrics: LogitDistributionMetrics,
    pub per_position: Vec<Fp8PositionQuality>,
    pub propagation: Vec<PropagationPointMetrics>,
    pub passes_quality_gate: bool,
    pub gate: &'static str,
}

#[derive(Debug, Serialize)]
pub struct Fp8QualityStudyReport {
    pub schema_version: u32,
    pub corpus_path: String,
    pub history_mode: &'static str,
    pub policies: Vec<Fp8PolicyQualityReport>,
    pub selected_policy_name: Option<String>,
    pub verdict: &'static str,
    pub greedy: Option<Fp8GreedyDiagnostics>,
}

#[derive(Debug, Serialize)]
pub struct Fp8GreedyPromptDiagnostic {
    pub prompt: String,
    pub bf16_text: String,
    pub candidate_text: String,
    pub bf16_tokens: usize,
    pub candidate_tokens: usize,
    pub first_divergent_token: Option<usize>,
    pub agreement_before_divergence: usize,
    pub exact_sequence_agreement: bool,
    pub output_length_agreement: bool,
}

#[derive(Debug, Serialize)]
pub struct Fp8GreedyDiagnostics {
    pub policy_name: String,
    pub prompts: usize,
    pub exact_sequence_agreement_rate: f64,
    pub output_length_agreement_rate: f64,
    pub diagnostics: Vec<Fp8GreedyPromptDiagnostic>,
}

#[derive(Debug, Serialize)]
pub struct Fp8SensitivitySiteReport {
    pub site: String,
    pub expected_decode_saving_us: f64,
    pub local_nrmse: f64,
    pub local_cosine: f64,
    pub final_hidden_nrmse: f64,
    pub final_hidden_cosine: f64,
    pub mean_logit_kl: f64,
    pub relative_nll_delta: f64,
    pub sensitivity_score: f64,
}

#[derive(Debug, Serialize)]
pub struct Fp8SensitivityReport {
    pub schema_version: u32,
    pub evaluation_sequences: usize,
    pub decode_positions_per_sequence: usize,
    pub sites: Vec<Fp8SensitivitySiteReport>,
}

#[derive(Debug, Serialize)]
pub struct Fp8PolicySearchStep {
    pub site: String,
    pub expected_decode_saving_us: f64,
    pub risk_score: f64,
    pub accepted: bool,
    pub enabled_sites_after_step: usize,
    pub relative_nll_delta: f64,
    pub mean_logit_kl: f64,
    pub final_hidden_nrmse: f64,
    pub final_hidden_cosine: f64,
}

#[derive(Debug, Serialize)]
pub struct Fp8PolicySearchReport {
    pub schema_version: u32,
    pub ranking: &'static str,
    pub fast_gate: &'static str,
    pub steps: Vec<Fp8PolicySearchStep>,
    pub selected_policy: Fp8PrecisionPolicy,
}

impl Engine {
    pub(crate) fn page_size(&self) -> usize {
        self.config.kv_page_size.value()
    }

    pub fn benchmark_hardware_profile(&self) -> Result<HardwareProfileBenchmarkReport> {
        let gpu_name = self.runtime.device_name()?;
        let (free_vram_bytes_before_benchmark, total_vram_bytes) = self.runtime.memory_info()?;
        let decode = self.benchmark_continuous_decode(
            &[1, 2, 4, 8, 16, 32, 64],
            &[128, 512, 2048, 8192],
            4,
            20,
        )?;
        let prefill =
            self.benchmark_prefill_chunks(&[32, 128, 512, 1024, 2048, 4096, 8192], 2, 5)?;
        let decode_points = decode
            .points
            .iter()
            .map(|point| CostPoint {
                batch: point.batch_size,
                tokens: 1,
                context: point.context_tokens,
                milliseconds: point.step_p95_ms,
            })
            .collect::<Vec<_>>();
        let prefill_points = prefill
            .iter()
            .map(|point| CostPoint {
                batch: 1,
                tokens: point.prompt_tokens,
                context: point.prompt_tokens,
                milliseconds: point.wall_p95_ms,
            })
            .collect::<Vec<_>>();
        let interactive_prompt_limit = prefill
            .iter()
            .filter(|point| point.wall_p95_ms < 350.0)
            .map(|point| point.prompt_tokens)
            .max()
            .unwrap_or(0);
        let cost_model = HardwareCostModel {
            schema_version: 1,
            gpu_name: gpu_name.clone(),
            page_size: self.config.kv_page_size.value(),
            decode_bf16: CostCurve::new(decode_points.clone())?,
            decode_fp8: CostCurve::new(decode_points)?,
            prefill_bf16: CostCurve::new(prefill_points)?,
            interactive_prompt_limit,
            ttft_slo_ms: 400.0,
            tpot_slo_ms: 50.0,
        };
        Ok(HardwareProfileBenchmarkReport {
            schema_version: 1,
            gpu_name,
            page_size: self.config.kv_page_size.value(),
            free_vram_bytes_before_benchmark,
            total_vram_bytes,
            interactive_ttft_headroom_ms: 50.0,
            decode,
            prefill,
            cost_model,
        })
    }

    pub fn benchmark_batched_fp8(
        &mut self,
        batch_sizes: &[usize],
        contexts: &[usize],
        warmup_pairs: usize,
        measured_pairs: usize,
    ) -> Result<BatchedFp8BenchmarkReport> {
        ensure!(
            self.model.decode_fp8_enabled(),
            "FP8 policy is not installed"
        );
        ensure!(!batch_sizes.is_empty(), "batched FP8 benchmark is empty");
        ensure!(measured_pairs > 0, "batched FP8 benchmark needs pairs");
        let maximum_batch = *batch_sizes.iter().max().context("missing batch size")?;
        self.model
            .prepare_batched_fp8(&self.runtime, maximum_batch)?;
        let mut points = Vec::with_capacity(batch_sizes.len() * contexts.len());
        for &context in contexts {
            for &batch in batch_sizes {
                let sequence_capacity = context
                    .checked_add(warmup_pairs)
                    .and_then(|value| value.checked_add(measured_pairs))
                    .context("batched FP8 sequence capacity overflow")?;
                ensure!(
                    sequence_capacity <= self.model.config().max_position_embeddings,
                    "batched FP8 benchmark exceeds model context"
                );
                let pages_per_sequence =
                    sequence_capacity.div_ceil(self.config.kv_page_size.value());
                let physical_pages = batch
                    .checked_mul(pages_per_sequence)
                    .context("batched FP8 KV page count overflow")?;
                let mut bf16_cache = self.model.new_batch_cache(
                    &self.runtime,
                    batch,
                    batch,
                    physical_pages,
                    self.config.kv_page_size,
                )?;
                let mut fp8_cache = self.model.new_batch_cache(
                    &self.runtime,
                    batch,
                    batch,
                    physical_pages,
                    self.config.kv_page_size,
                )?;
                for slot in 0..batch {
                    bf16_cache.reserve(slot, sequence_capacity)?;
                    fp8_cache.reserve(slot, sequence_capacity)?;
                }
                bf16_cache.prime_context(&self.runtime, batch, context)?;
                fp8_cache.prime_context(&self.runtime, batch, context)?;
                let mut token_ids = vec![42u32; batch];
                let mut positions = vec![0u32; batch];
                let request_slots = (0..batch)
                    .map(u32::try_from)
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                for pair in 0..warmup_pairs {
                    positions.fill(u32::try_from(context + pair)?);
                    token_ids.fill(42 + u32::try_from(pair % 11)?);
                    self.model.set_decode_fp8_enabled(false)?;
                    let _ = self.model.forward_decode_batch(
                        &self.runtime,
                        &mut bf16_cache,
                        &token_ids,
                        &positions,
                        &request_slots,
                    )?;
                    self.model.set_decode_fp8_enabled(true)?;
                    let _ = self.model.forward_decode_batch(
                        &self.runtime,
                        &mut fp8_cache,
                        &token_ids,
                        &positions,
                        &request_slots,
                    )?;
                }
                self.runtime.synchronize()?;
                let mut bf16_samples = Vec::with_capacity(measured_pairs);
                let mut fp8_samples = Vec::with_capacity(measured_pairs);
                let mut final_bf16 = None;
                let mut final_fp8 = None;
                for pair in 0..measured_pairs {
                    positions.fill(u32::try_from(context + warmup_pairs + pair)?);
                    token_ids.fill(53 + u32::try_from(pair % 13)?);
                    let bf16_first = pair % 2 == 0;
                    for fp8 in [!bf16_first, bf16_first] {
                        self.model.set_decode_fp8_enabled(fp8)?;
                        let started = self.runtime.record_timing_event()?;
                        let logits = if fp8 {
                            self.model.forward_decode_batch(
                                &self.runtime,
                                &mut fp8_cache,
                                &token_ids,
                                &positions,
                                &request_slots,
                            )?
                        } else {
                            self.model.forward_decode_batch(
                                &self.runtime,
                                &mut bf16_cache,
                                &token_ids,
                                &positions,
                                &request_slots,
                            )?
                        };
                        let finished = self.runtime.record_timing_event()?;
                        let elapsed = self.runtime.elapsed_ms(&started, &finished)?;
                        if fp8 {
                            fp8_samples.push(elapsed);
                            if pair + 1 == measured_pairs {
                                final_fp8 = Some(logits);
                            }
                        } else {
                            bf16_samples.push(elapsed);
                            if pair + 1 == measured_pairs {
                                final_bf16 = Some(logits);
                            }
                        }
                    }
                }
                let reference = self
                    .runtime
                    .download(final_bf16.as_ref().context("missing BF16 logits")?)?;
                let candidate = self
                    .runtime
                    .download(final_fp8.as_ref().context("missing FP8 logits")?)?;
                ensure!(reference.len() == candidate.len(), "logit size mismatch");
                let mut squared_error = 0.0f64;
                let mut reference_energy = 0.0f64;
                let mut candidate_energy = 0.0f64;
                let mut dot = 0.0f64;
                let mut non_finite_logits = 0usize;
                for (&left, &right) in reference.iter().zip(&candidate) {
                    let left = left.to_f32() as f64;
                    let right = right.to_f32() as f64;
                    if !left.is_finite() || !right.is_finite() {
                        non_finite_logits = non_finite_logits.saturating_add(1);
                        continue;
                    }
                    squared_error += (left - right) * (left - right);
                    reference_energy += left * left;
                    candidate_energy += right * right;
                    dot += left * right;
                }
                let logit_nrmse = (squared_error / reference_energy.max(f64::MIN_POSITIVE)).sqrt();
                let logit_cosine = dot
                    / (reference_energy * candidate_energy)
                        .sqrt()
                        .max(f64::MIN_POSITIVE);
                let vocab = self.model.config().vocab_size;
                let mut top1_agreement = 0usize;
                for row in 0..batch {
                    let range = row * vocab..(row + 1) * vocab;
                    if cpu_argmax_bf16(&reference[range.clone()])
                        == cpu_argmax_bf16(&candidate[range])
                    {
                        top1_agreement = top1_agreement.saturating_add(1);
                    }
                }
                let top1_agreement_ratio = top1_agreement as f64 / batch as f64;
                let bf16_mean_ms = mean(&bf16_samples).context("missing BF16 samples")?;
                let fp8_mean_ms = mean(&fp8_samples).context("missing FP8 samples")?;
                let smoke_quality_pass = non_finite_logits == 0
                    && logit_nrmse <= 0.10
                    && logit_cosine >= 0.995
                    && top1_agreement_ratio >= 0.99;
                let paired_speedup_mean = bf16_mean_ms / fp8_mean_ms;
                points.push(BatchedFp8Point {
                    batch_size: batch,
                    context_tokens: context,
                    warmup_pairs,
                    measured_pairs,
                    bf16_mean_ms,
                    bf16_p95_ms: percentile(&bf16_samples, 0.95).context("missing BF16 p95")?,
                    fp8_mean_ms,
                    fp8_p95_ms: percentile(&fp8_samples, 0.95).context("missing FP8 p95")?,
                    paired_speedup_mean,
                    logit_nrmse,
                    logit_cosine,
                    top1_agreement_ratio,
                    non_finite_logits,
                    smoke_quality_pass,
                    performance_pass: paired_speedup_mean > 1.02,
                });
            }
        }
        self.model.restrict_fp8_batch(1)?;
        self.model.set_decode_fp8_enabled(true)?;
        let all_smoke_quality_pass = points.iter().all(|point| point.smoke_quality_pass);
        let all_performance_pass = points.iter().all(|point| point.performance_pass);
        Ok(BatchedFp8BenchmarkReport {
            schema_version: 1,
            format: "tensorwide_e4m3",
            history_mode: "paired_teacher_forced_identical_tokens_independent_kv_caches",
            quality_scope: "synthetic_logit_smoke_only_not_checkpoint_corpus",
            promotion_gate: "not_promoted_without_disjoint_checkpoint_corpus_and_goodput_gain",
            points,
            all_smoke_quality_pass,
            all_performance_pass,
            promoted: false,
        })
    }

    fn benchmark_prefill_chunks(
        &self,
        prompt_lengths: &[usize],
        warmup_steps: usize,
        measured_steps: usize,
    ) -> Result<Vec<ServingPrefillPoint>> {
        ensure!(!prompt_lengths.is_empty(), "prefill profile is empty");
        ensure!(measured_steps > 0, "prefill profile needs samples");
        let mut points = Vec::with_capacity(prompt_lengths.len());
        for &tokens in prompt_lengths {
            ensure!(
                tokens > 0 && tokens <= self.model.config().max_position_embeddings,
                "invalid prefill profile length {tokens}"
            );
            let run_once = || -> Result<(f64, f64, f64)> {
                let pages = tokens.div_ceil(self.config.kv_page_size.value());
                let mut cache = self.model.new_batch_cache(
                    &self.runtime,
                    1,
                    tokens,
                    pages,
                    self.config.kv_page_size,
                )?;
                cache.reserve(0, tokens)?;
                let mut token_ids = vec![42u32; tokens];
                token_ids[0] = self.model.config().bos_token_id;
                let positions = (0..tokens)
                    .map(u32::try_from)
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let request_slots = vec![0u32; tokens];
                let segment_offsets = [0u32, u32::try_from(tokens)?];
                let segment_slots = [0u32];
                let output_rows = [u32::try_from(tokens - 1)?];
                let gpu_started = self.runtime.record_timing_event()?;
                let wall_started = Instant::now();
                let submit_started = Instant::now();
                let logits = self.model.forward_ragged_batch(
                    &self.runtime,
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
                let gpu_finished = self.runtime.record_timing_event()?;
                let submit_ms = elapsed_ms(submit_started);
                let gpu_ms = self.runtime.elapsed_ms(&gpu_started, &gpu_finished)?;
                let wall_ms = elapsed_ms(wall_started);
                drop(logits);
                Ok((wall_ms, gpu_ms, submit_ms))
            };
            for _ in 0..warmup_steps {
                let _ = run_once()?;
            }
            let pool_started = self.runtime.bf16_pool_stats();
            let mut wall = Vec::with_capacity(measured_steps);
            let mut gpu = Vec::with_capacity(measured_steps);
            let mut submit = Vec::with_capacity(measured_steps);
            for _ in 0..measured_steps {
                let (wall_ms, gpu_ms, submit_ms) = run_once()?;
                wall.push(wall_ms);
                gpu.push(gpu_ms);
                submit.push(submit_ms);
            }
            let pool_finished = self.runtime.bf16_pool_stats();
            let point = ServingPrefillPoint {
                prompt_tokens: tokens,
                warmup_steps,
                measured_steps,
                wall_mean_ms: mean(&wall).context("missing prefill wall samples")?,
                wall_p50_ms: percentile(&wall, 0.50).context("missing prefill wall p50")?,
                wall_p95_ms: percentile(&wall, 0.95).context("missing prefill wall p95")?,
                gpu_mean_ms: mean(&gpu).context("missing prefill GPU samples")?,
                gpu_p95_ms: percentile(&gpu, 0.95).context("missing prefill GPU p95")?,
                submit_cpu_mean_ms: mean(&submit).context("missing submit samples")?,
                bf16_pool_misses_after_warmup: pool_finished
                    .misses
                    .saturating_sub(pool_started.misses),
                bf16_pool_dropped_elements: pool_finished
                    .dropped_elements
                    .saturating_sub(pool_started.dropped_elements),
            };
            eprintln!(
                "prefill T={tokens}: wall_p95={:.3}ms gpu_p95={:.3}ms submit_mean={:.3}ms misses={}",
                point.wall_p95_ms,
                point.gpu_p95_ms,
                point.submit_cpu_mean_ms,
                point.bf16_pool_misses_after_warmup,
            );
            points.push(point);
        }
        Ok(points)
    }

    pub fn load(model_dir: &Path, device: usize, config: EngineConfig) -> Result<Self> {
        let runtime = CudaRuntime::new(device)?;
        let tokenizer = Lfm2Tokenizer::from_model_dir(model_dir)?;
        let model = Lfm2Model::load(&runtime, model_dir)?;
        Ok(Self {
            runtime,
            model,
            tokenizer,
            config,
            model_dir: model_dir.to_path_buf(),
        })
    }

    pub fn install_fp8_policy(&mut self, policy_path: &Path) -> Result<usize> {
        let policy = load_fp8_policy(policy_path)?;
        let enabled = self.model.install_fp8_policy(&self.runtime, &policy)?;
        ensure!(enabled > 0, "FP8 policy enables no runtime sites");
        Ok(enabled)
    }

    pub fn benchmark_continuous_decode(
        &self,
        batch_sizes: &[usize],
        contexts: &[usize],
        warmup_steps: usize,
        measured_steps: usize,
    ) -> Result<ServingDecodeBenchmarkReport> {
        ensure!(
            !batch_sizes.is_empty(),
            "serving benchmark needs batch sizes"
        );
        ensure!(!contexts.is_empty(), "serving benchmark needs contexts");
        ensure!(measured_steps > 0, "serving benchmark needs measured steps");
        let maximum_batch = *batch_sizes.iter().max().context("missing maximum batch")?;
        ensure!(maximum_batch > 0, "batch size must be positive");
        let maximum_context = *contexts.iter().max().context("missing maximum context")?;
        ensure!(
            maximum_context
                .checked_add(warmup_steps)
                .and_then(|value| value.checked_add(measured_steps))
                .context("serving benchmark sequence length overflow")?
                <= self.model.config().max_position_embeddings,
            "serving benchmark exceeds model context"
        );
        let ragged_prefill_validation = self.validate_ragged_prefill(32, 4)?;
        let maximum_sequence_tokens = maximum_context
            .checked_add(warmup_steps)
            .and_then(|value| value.checked_add(measured_steps))
            .context("serving benchmark capacity overflow")?;
        let maximum_pages = maximum_batch
            .checked_mul(maximum_sequence_tokens.div_ceil(self.config.kv_page_size.value()))
            .context("serving benchmark page count overflow")?;
        let mut cache = self.model.new_batch_cache(
            &self.runtime,
            maximum_batch,
            maximum_batch,
            maximum_pages,
            self.config.kv_page_size,
        )?;
        let (free_vram_bytes, _) = self.runtime.memory_info()?;
        let kv_bytes_per_token = self
            .model
            .config()
            .layer_types
            .iter()
            .filter(|kind| kind.as_str() == "full_attention")
            .count()
            .checked_mul(2)
            .and_then(|value| value.checked_mul(self.model.config().num_key_value_heads))
            .and_then(|value| value.checked_mul(self.model.config().head_dim()))
            .and_then(|value| value.checked_mul(std::mem::size_of::<half::bf16>()))
            .context("serving benchmark KV byte size overflow")?;
        let mut points = Vec::new();
        let mut skipped_capacity_points = Vec::new();
        for &context in contexts {
            for &batch in batch_sizes {
                let capacity = context
                    .checked_add(warmup_steps)
                    .and_then(|value| value.checked_add(measured_steps))
                    .context("serving decode capacity overflow")?;
                let required_kv_bytes = batch
                    .checked_mul(capacity)
                    .and_then(|value| value.checked_mul(kv_bytes_per_token))
                    .context("serving decode KV byte requirement overflow")?;
                if required_kv_bytes > free_vram_bytes {
                    skipped_capacity_points.push(ServingSkippedPoint {
                        batch_size: batch,
                        context_tokens: context,
                        required_kv_bytes,
                        free_vram_bytes,
                        reason: "insufficient_free_vram_before_kv_allocation",
                    });
                    continue;
                }
                for slot in 0..batch {
                    cache.reserve(slot, capacity)?;
                }
                cache.prime_context(&self.runtime, batch, context)?;
                let request_slots = (0..batch)
                    .map(u32::try_from)
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let mut token_ids = vec![42u32; batch];
                let mut positions = vec![u32::try_from(context)?; batch];
                for step in 0..warmup_steps {
                    token_ids.fill(42 + u32::try_from(step % 17)?);
                    positions.fill(u32::try_from(context + step)?);
                    let logits = self.model.forward_decode_batch(
                        &self.runtime,
                        &mut cache,
                        &token_ids,
                        &positions,
                        &request_slots,
                    )?;
                    drop(logits);
                }
                self.runtime.synchronize()?;
                let transfers_started = cache.transfers();
                let bf16_started = self.runtime.bf16_pool_stats();
                let fp8_started = self.runtime.fp8_pool_stats();
                let mut samples = Vec::with_capacity(measured_steps);
                let mut final_logits = None;
                for step in 0..measured_steps {
                    token_ids.fill(59 + u32::try_from(step % 23)?);
                    positions.fill(u32::try_from(context + warmup_steps + step)?);
                    let started = self.runtime.record_timing_event()?;
                    let logits = self.model.forward_decode_batch(
                        &self.runtime,
                        &mut cache,
                        &token_ids,
                        &positions,
                        &request_slots,
                    )?;
                    let finished = self.runtime.record_timing_event()?;
                    samples.push(self.runtime.elapsed_ms(&started, &finished)?);
                    final_logits = Some(logits);
                }
                let transfers_finished = cache.transfers();
                let bf16_finished = self.runtime.bf16_pool_stats();
                let fp8_finished = self.runtime.fp8_pool_stats();
                let step_mean_ms = mean(&samples).context("missing serving decode mean")?;
                let step_p50_ms = percentile(&samples, 0.50).context("missing serving p50")?;
                let step_p95_ms = percentile(&samples, 0.95).context("missing serving p95")?;
                let output_tokens_per_second = batch as f64 * 1000.0 / step_mean_ms;
                let tpot_slo_pass = step_p95_ms < 50.0;
                let snapshot = cache.kv_snapshot();
                let live_tokens = batch * (context + warmup_steps + measured_steps);
                let physical_token_capacity =
                    snapshot.allocated_pages * self.config.kv_page_size.value();
                let fragmentation = if physical_token_capacity == 0 {
                    0.0
                } else {
                    (physical_token_capacity.saturating_sub(live_tokens)) as f64
                        / physical_token_capacity as f64
                };
                let output_tokens = batch
                    .checked_mul(measured_steps)
                    .context("output token count overflow")?;
                let logits_host = self
                    .runtime
                    .download(final_logits.as_ref().context("missing final logits")?)?;
                let vocab = self.model.config().vocab_size;
                let reference = &logits_host[..vocab];
                let reference_top1 = reference
                    .iter()
                    .enumerate()
                    .max_by(|left, right| left.1.to_f32().total_cmp(&right.1.to_f32()))
                    .map(|(index, _)| index)
                    .context("empty reference logits")?;
                let reference_energy = reference
                    .iter()
                    .map(|value| {
                        let value = value.to_f32() as f64;
                        value * value
                    })
                    .sum::<f64>()
                    .max(f64::MIN_POSITIVE);
                let mut row_nrmse_max = 0.0f64;
                let mut row_top1_agreement = true;
                for row in 1..batch {
                    let candidate = &logits_host[row * vocab..(row + 1) * vocab];
                    let squared_error = reference
                        .iter()
                        .zip(candidate)
                        .map(|(reference, candidate)| {
                            let difference = reference.to_f32() as f64 - candidate.to_f32() as f64;
                            difference * difference
                        })
                        .sum::<f64>();
                    row_nrmse_max = row_nrmse_max.max((squared_error / reference_energy).sqrt());
                    let candidate_top1 = candidate
                        .iter()
                        .enumerate()
                        .max_by(|left, right| left.1.to_f32().total_cmp(&right.1.to_f32()))
                        .map(|(index, _)| index)
                        .context("empty candidate logits")?;
                    row_top1_agreement &= candidate_top1 == reference_top1;
                }
                points.push(ServingDecodePoint {
                    batch_size: batch,
                    context_tokens: context,
                    precision: if self.model.decode_fp8_enabled()
                        && batch <= self.model.maximum_fp8_batch()
                    {
                        "selective_e4m3"
                    } else {
                        "bf16"
                    },
                    warmup_steps,
                    measured_steps,
                    step_mean_ms,
                    step_p50_ms,
                    step_p95_ms,
                    output_tokens_per_second,
                    goodput_tokens_per_second: if tpot_slo_pass {
                        output_tokens_per_second
                    } else {
                        0.0
                    },
                    tpot_slo_pass,
                    kv_pages_allocated: snapshot.allocated_pages,
                    kv_internal_fragmentation_ratio: fragmentation,
                    h2d_bytes_per_output_token: transfers_finished
                        .h2d_bytes
                        .saturating_sub(transfers_started.h2d_bytes)
                        as f64
                        / output_tokens as f64,
                    h2d_calls_per_step: transfers_finished
                        .h2d_calls
                        .saturating_sub(transfers_started.h2d_calls)
                        as f64
                        / measured_steps as f64,
                    bf16_pool_misses_after_warmup: bf16_finished
                        .misses
                        .saturating_sub(bf16_started.misses),
                    fp8_pool_misses_after_warmup: fp8_finished
                        .misses
                        .saturating_sub(fp8_started.misses),
                    identical_sequence_row_nrmse_max: row_nrmse_max,
                    identical_sequence_top1_agreement: row_top1_agreement,
                });
                eprintln!(
                    "serving decode B={batch} ctx={context}: mean={step_mean_ms:.3}ms p95={step_p95_ms:.3}ms tok/s={output_tokens_per_second:.1}"
                );
            }
        }
        Ok(ServingDecodeBenchmarkReport {
            schema_version: 1,
            design: "persistent_shared_kv_ragged_true_m_equals_batch_fixed_token_decode",
            page_size: self.config.kv_page_size.value(),
            tpot_slo_ms: 50.0,
            ragged_prefill_validation,
            points,
            skipped_capacity_points,
        })
    }

    fn validate_ragged_prefill(
        &self,
        tokens_per_sequence: usize,
        batch_size: usize,
    ) -> Result<RaggedPrefillValidation> {
        ensure!(
            tokens_per_sequence > 0 && batch_size > 1,
            "invalid ragged validation shape"
        );
        let mut sequence = vec![42u32; tokens_per_sequence];
        sequence[0] = self.model.config().bos_token_id;
        let mut legacy_cache =
            self.model
                .new_cache(&self.runtime, tokens_per_sequence, self.config.kv_page_size)?;
        let legacy_started = self.runtime.record_timing_event()?;
        let legacy_logits =
            self.model
                .forward_logits(&self.runtime, &mut legacy_cache, &sequence)?;
        let legacy_finished = self.runtime.record_timing_event()?;
        let legacy_ms = self.runtime.elapsed_ms(&legacy_started, &legacy_finished)?;
        let pages = batch_size
            .checked_mul(tokens_per_sequence.div_ceil(self.config.kv_page_size.value()))
            .context("ragged validation page overflow")?;
        let flattened_tokens = batch_size
            .checked_mul(tokens_per_sequence)
            .context("ragged validation token overflow")?;
        let mut cache = self.model.new_batch_cache(
            &self.runtime,
            batch_size,
            flattened_tokens,
            pages,
            self.config.kv_page_size,
        )?;
        for slot in 0..batch_size {
            cache.reserve(slot, tokens_per_sequence)?;
        }
        let mut tokens = Vec::with_capacity(flattened_tokens);
        let mut positions = Vec::with_capacity(flattened_tokens);
        let mut request_slots = Vec::with_capacity(flattened_tokens);
        let mut segment_offsets = Vec::with_capacity(batch_size + 1);
        let mut segment_slots = Vec::with_capacity(batch_size);
        let mut output_rows = Vec::with_capacity(batch_size);
        segment_offsets.push(0u32);
        for slot in 0..batch_size {
            tokens.extend_from_slice(&sequence);
            positions.extend(
                (0..tokens_per_sequence)
                    .map(u32::try_from)
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            );
            request_slots.extend(std::iter::repeat_n(
                u32::try_from(slot)?,
                tokens_per_sequence,
            ));
            segment_offsets.push(u32::try_from(tokens.len())?);
            segment_slots.push(u32::try_from(slot)?);
            output_rows.push(u32::try_from(tokens.len() - 1)?);
        }
        let ragged_started = self.runtime.record_timing_event()?;
        let ragged_logits = self.model.forward_ragged_batch(
            &self.runtime,
            &mut cache,
            RaggedBatchInput {
                token_ids: &tokens,
                positions: &positions,
                request_slots: &request_slots,
                segment_offsets: &segment_offsets,
                segment_slots: &segment_slots,
                output_rows: &output_rows,
            },
        )?;
        let ragged_finished = self.runtime.record_timing_event()?;
        let ragged_ms = self.runtime.elapsed_ms(&ragged_started, &ragged_finished)?;
        let reference = self.runtime.download(&legacy_logits)?;
        let candidate = self.runtime.download(&ragged_logits)?;
        let mut nrmse_max = 0.0f64;
        let mut cosine_min = 1.0f64;
        let reference_top1 = cpu_argmax_bf16(&reference).context("empty legacy logits")?;
        let mut top1_agreement = true;
        for row in 0..batch_size {
            let values = &candidate[row * reference.len()..(row + 1) * reference.len()];
            let mut squared_error = 0.0f64;
            let mut reference_energy = 0.0f64;
            let mut candidate_energy = 0.0f64;
            let mut dot = 0.0f64;
            for (left, right) in reference.iter().zip(values) {
                let left = left.to_f32() as f64;
                let right = right.to_f32() as f64;
                squared_error += (left - right) * (left - right);
                reference_energy += left * left;
                candidate_energy += right * right;
                dot += left * right;
            }
            nrmse_max =
                nrmse_max.max((squared_error / reference_energy.max(f64::MIN_POSITIVE)).sqrt());
            cosine_min = cosine_min.min(
                dot / (reference_energy * candidate_energy)
                    .sqrt()
                    .max(f64::MIN_POSITIVE),
            );
            top1_agreement &= cpu_argmax_bf16(values) == Some(reference_top1);
        }
        Ok(RaggedPrefillValidation {
            tokens_per_sequence,
            batch_size,
            legacy_contiguous_ms: legacy_ms,
            ragged_paged_ms: ragged_ms,
            final_logits_nrmse_max: nrmse_max,
            final_logits_cosine_min: cosine_min,
            top1_agreement,
        })
    }

    pub fn validate_fp8_policy(
        &mut self,
        policy_path: &Path,
        corpus_path: &Path,
        sequences: usize,
        max_tokens: usize,
    ) -> Result<Fp8FinalValidationReport> {
        ensure!(sequences > 0, "FP8 validation requires sequences");
        ensure!(
            (64..=self.model.config().max_position_embeddings).contains(&max_tokens),
            "FP8 validation token limit must be in [64, {}]",
            self.model.config().max_position_embeddings
        );
        let policy = load_fp8_policy(policy_path)?;
        let file = File::open(corpus_path).with_context(|| {
            format!(
                "failed to open FP8 validation corpus {}",
                corpus_path.display()
            )
        })?;
        let token_sequences = calibration_sequences(
            BufReader::new(file),
            &self.tokenizer,
            self.model.config().bos_token_id,
            sequences,
            max_tokens,
        )?;
        ensure!(
            token_sequences.len() == sequences,
            "validation corpus yielded {} sequences, need {sequences}",
            token_sequences.len()
        );
        let quality = self.evaluate_fp8_policy(&policy, &token_sequences)?;
        let greedy = self.run_fp8_greedy_diagnostics(&policy)?;
        Ok(Fp8FinalValidationReport {
            schema_version: 1,
            corpus_path: corpus_path.display().to_string(),
            history_mode: "teacher_forced_identical_tokens_bf16_vs_candidate",
            quality,
            greedy,
        })
    }

    pub fn benchmark_installed_fp8(
        &mut self,
        requested_contexts: &[usize],
        completion_tokens: usize,
        warmup_pairs: usize,
        measured_pairs: usize,
    ) -> Result<Fp8BenchmarkReport> {
        ensure!(
            completion_tokens >= 2,
            "benchmark requires at least two generated tokens"
        );
        ensure!(measured_pairs > 0, "benchmark requires measured pairs");
        let options = GenerationOptions {
            max_new_tokens: completion_tokens,
            sampling: SamplingConfig {
                temperature: 0.0,
                ..SamplingConfig::default()
            },
        };
        let mut workloads = Vec::with_capacity(requested_contexts.len());
        for &requested_context_tokens in requested_contexts {
            let prompt = self.benchmark_prompt(requested_context_tokens, completion_tokens)?;
            let actual_prompt_tokens = self.tokenizer.encode_user_prompt(&prompt)?.len();
            eprintln!(
                "benchmarking BF16/FP8 context requested={} actual={} pairs={}",
                requested_context_tokens, actual_prompt_tokens, measured_pairs
            );
            for _ in 0..warmup_pairs {
                self.model.set_decode_fp8_enabled(false)?;
                let _reference = self.generate_fixed_steps(&prompt, options)?;
                self.model.set_decode_fp8_enabled(true)?;
                let _candidate = self.generate_fixed_steps(&prompt, options)?;
            }
            let mut bf16 = Vec::with_capacity(measured_pairs);
            let mut fp8 = Vec::with_capacity(measured_pairs);
            let mut paired_tpot_speedups = Vec::with_capacity(measured_pairs);
            let mut paired_ttft_ratios = Vec::with_capacity(measured_pairs);
            for pair in 0..measured_pairs {
                let fp8_first = pair % 2 == 1;
                let (reference, candidate) = if fp8_first {
                    self.model.set_decode_fp8_enabled(true)?;
                    let candidate = self.generate_fixed_steps(&prompt, options)?;
                    self.model.set_decode_fp8_enabled(false)?;
                    let reference = self.generate_fixed_steps(&prompt, options)?;
                    (reference, candidate)
                } else {
                    self.model.set_decode_fp8_enabled(false)?;
                    let reference = self.generate_fixed_steps(&prompt, options)?;
                    self.model.set_decode_fp8_enabled(true)?;
                    let candidate = self.generate_fixed_steps(&prompt, options)?;
                    (reference, candidate)
                };
                ensure!(
                    reference.completion_tokens == completion_tokens
                        && candidate.completion_tokens == completion_tokens,
                    "benchmark generation stopped early: BF16={} FP8={} requested={}",
                    reference.completion_tokens,
                    candidate.completion_tokens,
                    completion_tokens
                );
                let reference_tpot = reference
                    .metrics
                    .tpot_mean_ms
                    .context("missing BF16 benchmark TPOT")?;
                let candidate_tpot = candidate
                    .metrics
                    .tpot_mean_ms
                    .context("missing FP8 benchmark TPOT")?;
                paired_tpot_speedups.push(reference_tpot / candidate_tpot);
                paired_ttft_ratios.push(reference.metrics.ttft_ms / candidate.metrics.ttft_ms);
                bf16.push(reference.metrics);
                fp8.push(candidate.metrics);
            }
            self.model.set_decode_fp8_enabled(false)?;
            workloads.push(Fp8BenchmarkWorkload {
                requested_context_tokens,
                actual_prompt_tokens,
                completion_tokens,
                warmup_pairs,
                measured_pairs,
                bf16: summarize_generation_metrics(&bf16)?,
                fp8: summarize_generation_metrics(&fp8)?,
                paired_tpot_speedup_mean: mean(&paired_tpot_speedups)
                    .context("missing paired TPOT speedup")?,
                paired_tpot_speedup_p50: percentile(&paired_tpot_speedups, 0.50)
                    .context("missing paired TPOT p50 speedup")?,
                paired_tpot_speedup_p95: percentile(&paired_tpot_speedups, 0.95)
                    .context("missing paired TPOT p95 speedup")?,
                paired_tpot_speedup_min: paired_tpot_speedups
                    .iter()
                    .copied()
                    .min_by(f64::total_cmp)
                    .context("missing paired TPOT minimum")?,
                paired_tpot_speedup_max: paired_tpot_speedups
                    .iter()
                    .copied()
                    .max_by(f64::total_cmp)
                    .context("missing paired TPOT maximum")?,
                paired_ttft_ratio_mean: mean(&paired_ttft_ratios)
                    .context("missing paired TTFT ratio")?,
            });
        }
        Ok(Fp8BenchmarkReport {
            schema_version: 1,
            design: "same_process_interleaved_order_balanced_bf16_vs_decode_only_fp8",
            warmup_pairs,
            measured_pairs,
            completion_tokens,
            workloads,
        })
    }

    fn benchmark_prompt(&self, target_tokens: usize, completion_tokens: usize) -> Result<String> {
        ensure!(target_tokens > 0, "benchmark context must be positive");
        let unit =
            "A careful systems benchmark reports workload, hardware, method, and uncertainty. ";
        let suffix = "\nWrite the integers from 1 to 1000, separated by commas.";
        let mut repetitions = 1usize;
        let mut prompt = format!("{unit}{suffix}");
        loop {
            let actual = self.tokenizer.encode_user_prompt(&prompt)?.len();
            if actual >= target_tokens {
                ensure!(
                    actual
                        .checked_add(completion_tokens)
                        .context("benchmark sequence length overflow")?
                        <= self.model.config().max_position_embeddings,
                    "benchmark context {} plus completion {} exceeds model limit {}",
                    actual,
                    completion_tokens,
                    self.model.config().max_position_embeddings
                );
                return Ok(prompt);
            }
            let missing_ratio = target_tokens as f64 / actual.max(1) as f64;
            let next = ((repetitions as f64 * missing_ratio).ceil() as usize)
                .max(repetitions.saturating_add(1));
            repetitions = next;
            prompt = format!("{}{suffix}", unit.repeat(repetitions));
        }
    }

    pub fn calibrate_fp8(
        &mut self,
        corpus_path: &Path,
        evaluation_corpus_path: &Path,
        max_sequences: usize,
        max_sequence_tokens: usize,
        evaluation_sequences: usize,
        evaluation_max_tokens: usize,
    ) -> Result<Fp8CalibrationArtifacts> {
        ensure!(
            max_sequences > 0,
            "calibration max sequences must be positive"
        );
        ensure!(
            max_sequence_tokens > 0,
            "calibration max sequence tokens must be positive"
        );
        ensure!(
            max_sequence_tokens <= self.model.config().max_position_embeddings,
            "calibration max sequence tokens {} exceeds model limit {}",
            max_sequence_tokens,
            self.model.config().max_position_embeddings
        );
        ensure!(
            evaluation_sequences > 0,
            "FP8 evaluation requires sequences"
        );
        ensure!(
            (64..=self.model.config().max_position_embeddings).contains(&evaluation_max_tokens),
            "FP8 evaluation token limit must be in [64, {}]",
            self.model.config().max_position_embeddings
        );

        let file = File::open(corpus_path).with_context(|| {
            format!(
                "failed to open calibration corpus {}",
                corpus_path.display()
            )
        })?;
        let reader = BufReader::new(file);
        let sequences = calibration_sequences(
            reader,
            &self.tokenizer,
            self.model.config().bos_token_id,
            max_sequences,
            max_sequence_tokens,
        )?;
        ensure!(
            sequences.len() == max_sequences,
            "calibration corpus yielded {} sequences, need {max_sequences}",
            sequences.len()
        );
        let mut collector = CalibrationCollector::new(64);

        eprintln!("collecting checkpoint weight statistics");
        self.model
            .collect_calibration_weights(&self.runtime, &mut collector)?;

        for token_ids in &sequences {
            collector.set_activation_phase(CalibrationPhase::Prefill)?;
            let mut prefill_cache =
                self.model
                    .new_cache(&self.runtime, token_ids.len(), self.config.kv_page_size)?;
            let _logits = self.model.forward_logits_calibrated(
                &self.runtime,
                &mut prefill_cache,
                token_ids,
                &mut collector,
            )?;
            collector.record_prefill_forward();

            let positions = decode_sample_positions(token_ids.len(), 8);
            let mut decode_cache =
                self.model
                    .new_cache(&self.runtime, token_ids.len(), self.config.kv_page_size)?;
            let mut cursor = 0usize;
            for position in positions {
                if cursor < position {
                    let _logits = self.model.forward_logits(
                        &self.runtime,
                        &mut decode_cache,
                        &token_ids[cursor..position],
                    )?;
                }
                collector.set_activation_phase(CalibrationPhase::Decode)?;
                let _logits = self.model.forward_logits_calibrated(
                    &self.runtime,
                    &mut decode_cache,
                    &token_ids[position..position + 1],
                    &mut collector,
                )?;
                cursor = position + 1;
                collector.record_decode_forward(cursor)?;
            }
            collector.record_sequence(token_ids.len())?;
            eprintln!(
                "calibrated sequence {}/{} ({} tokens, {} decode M=1 samples)",
                collector.sequence_count(),
                max_sequences,
                token_ids.len(),
                decode_sample_positions(token_ids.len(), 8).len(),
            );
        }

        let gemm_error = self
            .model
            .characterize_calibration_gemms(&self.runtime, &collector)?;
        let calibration = collector.finish(
            self.model_dir.display().to_string(),
            corpus_path.display().to_string(),
            max_sequences,
            max_sequence_tokens,
            self.config.kv_page_size.value(),
        )?;
        let evaluation_file = File::open(evaluation_corpus_path).with_context(|| {
            format!(
                "failed to open FP8 evaluation corpus {}",
                evaluation_corpus_path.display()
            )
        })?;
        let evaluation_token_sequences = calibration_sequences(
            BufReader::new(evaluation_file),
            &self.tokenizer,
            self.model.config().bos_token_id,
            evaluation_sequences,
            evaluation_max_tokens,
        )?;
        ensure!(
            evaluation_token_sequences.len() == evaluation_sequences,
            "evaluation corpus yielded {} sequences, need {evaluation_sequences}",
            evaluation_token_sequences.len()
        );

        let base_policy = Fp8PrecisionPolicy::from_local_errors(&gemm_error, "local_safe_all");
        let sensitivity = self.run_fp8_sensitivity(&base_policy, &evaluation_token_sequences)?;
        let policy_search =
            self.search_fp8_policy(&base_policy, &sensitivity, &evaluation_token_sequences)?;
        let mut policies = Vec::with_capacity(13);
        for (name, class) in [
            ("policy_a_mlp", PrecisionClass::Mlp),
            ("policy_b_mlp_lm_head", PrecisionClass::MlpLmHead),
            ("policy_c_mlp_lm_head_conv", PrecisionClass::MlpLmHeadConv),
            ("policy_d_maximal_local_safe", PrecisionClass::All),
        ] {
            let mut policy = base_policy.clone();
            policy.name = name.to_string();
            policy.retain_class(class);
            policies.push(policy);
        }
        let accepted_order: Vec<_> = policy_search
            .steps
            .iter()
            .filter(|step| step.accepted)
            .map(|step| step.site.as_str())
            .collect();
        for requested in [1usize, 2, 4, 8, 12, 16, 24, 32] {
            if requested > accepted_order.len() {
                continue;
            }
            let mut policy = policy_search.selected_policy.clone();
            policy.name = format!("policy_frontier_{requested}");
            for site in &mut policy.sites {
                site.enabled = accepted_order[..requested].contains(&site.site.as_str());
            }
            policies.push(policy);
        }
        policies.push(policy_search.selected_policy.clone());

        let mut quality_reports = Vec::with_capacity(policies.len());
        for policy in &policies {
            eprintln!("evaluating FP8 quality policy {}", policy.name);
            quality_reports.push(self.evaluate_fp8_policy(policy, &evaluation_token_sequences)?);
        }
        let selected_policy = quality_reports
            .iter()
            .filter(|report| report.passes_quality_gate)
            .filter_map(|report| {
                let policy = policies
                    .iter()
                    .find(|policy| policy.name == report.policy_name)?;
                let expected_saving = policy
                    .sites
                    .iter()
                    .filter(|site| site.enabled)
                    .map(|site| site.expected_decode_saving_us)
                    .sum::<f64>();
                Some((expected_saving, policy))
            })
            .max_by(|left, right| left.0.total_cmp(&right.0))
            .map(|(_, policy)| policy.clone());
        let greedy = match selected_policy.as_ref() {
            Some(policy) => Some(self.run_fp8_greedy_diagnostics(policy)?),
            None => None,
        };
        let quality = Fp8QualityStudyReport {
            schema_version: 1,
            corpus_path: evaluation_corpus_path.display().to_string(),
            history_mode: "teacher_forced_identical_tokens_bf16_vs_candidate",
            policies: quality_reports,
            selected_policy_name: selected_policy.as_ref().map(|policy| policy.name.clone()),
            verdict: if selected_policy.is_some() {
                "quality_gate_pass"
            } else {
                "quality_gate_reject_all"
            },
            greedy,
        };
        Ok(Fp8CalibrationArtifacts {
            calibration,
            gemm_error,
            policies,
            quality,
            sensitivity,
            policy_search,
            selected_policy,
        })
    }

    fn evaluate_fp8_policy(
        &mut self,
        policy: &Fp8PrecisionPolicy,
        sequences: &[Vec<u32>],
    ) -> Result<Fp8PolicyQualityReport> {
        let enabled_sites = self.model.install_fp8_policy(&self.runtime, policy)?;
        ensure!(enabled_sites > 0, "FP8 policy enables no sites");
        let mut logits = LogitMetricAccumulator::default();
        let mut per_position = BTreeMap::<usize, LogitMetricAccumulator>::new();
        let mut propagation = PropagationAccumulator::default();
        let mut evaluation_source_tokens = 0usize;

        for (sequence_index, tokens) in sequences.iter().enumerate() {
            ensure!(tokens.len() >= 2, "evaluation sequence is too short");
            evaluation_source_tokens = evaluation_source_tokens
                .checked_add(tokens.len())
                .context("evaluation token count overflow")?;
            let mut reference_cache =
                self.model
                    .new_cache(&self.runtime, tokens.len(), self.config.kv_page_size)?;
            let mut candidate_cache =
                self.model
                    .new_cache(&self.runtime, tokens.len(), self.config.kv_page_size)?;
            let capture_positions = decode_sample_positions(tokens.len() - 1, 8);

            for input_position in 0..tokens.len() - 1 {
                let capture = capture_positions.contains(&input_position);
                self.model.set_decode_fp8_enabled(false)?;
                let (reference_logits, reference_capture) = if capture {
                    let mut capture = HiddenCapture::default();
                    let logits = self.model.forward_logits_captured(
                        &self.runtime,
                        &mut reference_cache,
                        &tokens[input_position..input_position + 1],
                        &mut capture,
                    )?;
                    (logits, Some(capture))
                } else {
                    (
                        self.model.forward_logits(
                            &self.runtime,
                            &mut reference_cache,
                            &tokens[input_position..input_position + 1],
                        )?,
                        None,
                    )
                };

                self.model.set_decode_fp8_enabled(true)?;
                let (candidate_logits, candidate_capture) = if capture {
                    let mut capture = HiddenCapture::default();
                    let logits = self.model.forward_logits_captured(
                        &self.runtime,
                        &mut candidate_cache,
                        &tokens[input_position..input_position + 1],
                        &mut capture,
                    )?;
                    (logits, Some(capture))
                } else {
                    (
                        self.model.forward_logits(
                            &self.runtime,
                            &mut candidate_cache,
                            &tokens[input_position..input_position + 1],
                        )?,
                        None,
                    )
                };
                let reference_host = self.runtime.download(&reference_logits)?;
                let candidate_host = self.runtime.download(&candidate_logits)?;
                let target = tokens[input_position + 1];
                logits.observe(&reference_host, &candidate_host, target)?;
                per_position.entry(input_position).or_default().observe(
                    &reference_host,
                    &candidate_host,
                    target,
                )?;
                if let (Some(reference), Some(candidate)) =
                    (reference_capture.as_ref(), candidate_capture.as_ref())
                {
                    propagation.observe(reference, candidate)?;
                }
            }
            eprintln!(
                "evaluated {} sequence {}/{}",
                policy.name,
                sequence_index + 1,
                sequences.len()
            );
        }
        self.model.set_decode_fp8_enabled(false)?;
        let metrics = logits.finish()?;
        let per_position = per_position
            .into_iter()
            .map(|(input_position, accumulator)| {
                Ok(Fp8PositionQuality {
                    input_position,
                    metrics: accumulator.finish()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let propagation = propagation.finish();
        let catastrophic_hidden_drift = propagation
            .iter()
            .any(|point| point.non_finite_values > 0 || point.nrmse > 0.10 || point.cosine < 0.99);
        let passes_quality_gate = metrics.non_finite_logits == 0
            && metrics.relative_nll_delta <= 0.01
            && !catastrophic_hidden_drift;
        Ok(Fp8PolicyQualityReport {
            policy_name: policy.name.clone(),
            enabled_sites,
            evaluation_sequences: sequences.len(),
            evaluation_source_tokens,
            metrics,
            per_position,
            propagation,
            passes_quality_gate,
            gate: "relative_nll_delta<=1%_and_no_nonfinite_and_hidden_cosine>=0.99_and_hidden_nrmse<=0.10",
        })
    }

    fn evaluate_fp8_policy_sampled(
        &mut self,
        policy: &Fp8PrecisionPolicy,
        sequences: &[Vec<u32>],
        positions_per_sequence: usize,
    ) -> Result<Fp8PolicyQualityReport> {
        let enabled_sites = self.model.install_fp8_policy(&self.runtime, policy)?;
        ensure!(enabled_sites > 0, "sampled FP8 policy enables no sites");
        let mut logits = LogitMetricAccumulator::default();
        let mut per_position = BTreeMap::<usize, LogitMetricAccumulator>::new();
        let mut propagation = PropagationAccumulator::default();
        let mut evaluation_source_tokens = 0usize;

        for tokens in sequences {
            ensure!(
                tokens.len() >= 3,
                "sampled evaluation sequence is too short"
            );
            evaluation_source_tokens = evaluation_source_tokens
                .checked_add(tokens.len())
                .context("sampled evaluation token count overflow")?;
            let mut reference_cache =
                self.model
                    .new_cache(&self.runtime, tokens.len(), self.config.kv_page_size)?;
            let mut candidate_cache =
                self.model
                    .new_cache(&self.runtime, tokens.len(), self.config.kv_page_size)?;
            let positions = decode_sample_positions(tokens.len() - 1, positions_per_sequence);
            let mut cursor = 0usize;
            for position in positions {
                if cursor < position {
                    self.model.set_decode_fp8_enabled(false)?;
                    let _reference = self.model.forward_logits(
                        &self.runtime,
                        &mut reference_cache,
                        &tokens[cursor..position],
                    )?;
                    let _candidate = self.model.forward_logits(
                        &self.runtime,
                        &mut candidate_cache,
                        &tokens[cursor..position],
                    )?;
                }

                self.model.set_decode_fp8_enabled(false)?;
                let mut reference_capture = HiddenCapture::default();
                let reference_logits = self.model.forward_logits_captured(
                    &self.runtime,
                    &mut reference_cache,
                    &tokens[position..position + 1],
                    &mut reference_capture,
                )?;
                self.model.set_decode_fp8_enabled(true)?;
                let mut candidate_capture = HiddenCapture::default();
                let candidate_logits = self.model.forward_logits_captured(
                    &self.runtime,
                    &mut candidate_cache,
                    &tokens[position..position + 1],
                    &mut candidate_capture,
                )?;
                let reference_host = self.runtime.download(&reference_logits)?;
                let candidate_host = self.runtime.download(&candidate_logits)?;
                let target = tokens[position + 1];
                logits.observe(&reference_host, &candidate_host, target)?;
                per_position.entry(position).or_default().observe(
                    &reference_host,
                    &candidate_host,
                    target,
                )?;
                propagation.observe(&reference_capture, &candidate_capture)?;
                cursor = position + 1;
            }
        }
        self.model.set_decode_fp8_enabled(false)?;
        let metrics = logits.finish()?;
        let per_position = per_position
            .into_iter()
            .map(|(input_position, accumulator)| {
                Ok(Fp8PositionQuality {
                    input_position,
                    metrics: accumulator.finish()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let propagation = propagation.finish();
        let catastrophic_hidden_drift = propagation
            .iter()
            .any(|point| point.non_finite_values > 0 || point.nrmse > 0.10 || point.cosine < 0.99);
        let passes_quality_gate = metrics.non_finite_logits == 0
            && metrics.relative_nll_delta <= 0.01
            && !catastrophic_hidden_drift;
        Ok(Fp8PolicyQualityReport {
            policy_name: policy.name.clone(),
            enabled_sites,
            evaluation_sequences: sequences.len(),
            evaluation_source_tokens,
            metrics,
            per_position,
            propagation,
            passes_quality_gate,
            gate: "sampled_proxy_only_not_final_quality_gate",
        })
    }

    fn run_fp8_sensitivity(
        &mut self,
        base_policy: &Fp8PrecisionPolicy,
        sequences: &[Vec<u32>],
    ) -> Result<Fp8SensitivityReport> {
        let subset = &sequences[..sequences.len().min(4)];
        let mut sites = Vec::with_capacity(base_policy.sites.len());
        for index in 0..base_policy.sites.len() {
            let mut policy = base_policy.clone();
            policy.name = format!("single_site_{}", policy.sites[index].site);
            for site in &mut policy.sites {
                site.enabled = false;
            }
            policy.sites[index].enabled = true;
            let quality = self.evaluate_fp8_policy_sampled(&policy, subset, 4)?;
            let final_hidden = quality
                .propagation
                .iter()
                .find(|point| point.point == "final_rms_norm")
                .context("single-site propagation missing final RMSNorm")?;
            let source = &base_policy.sites[index];
            let sensitivity_score = final_hidden.nrmse
                + quality.metrics.mean_kl_bf16_to_candidate
                + quality.metrics.relative_nll_delta.max(0.0);
            eprintln!(
                "single-site sensitivity {} score={:.6}",
                source.site, sensitivity_score
            );
            sites.push(Fp8SensitivitySiteReport {
                site: source.site.clone(),
                expected_decode_saving_us: source.expected_decode_saving_us,
                local_nrmse: source.local_nrmse,
                local_cosine: source.local_cosine,
                final_hidden_nrmse: final_hidden.nrmse,
                final_hidden_cosine: final_hidden.cosine,
                mean_logit_kl: quality.metrics.mean_kl_bf16_to_candidate,
                relative_nll_delta: quality.metrics.relative_nll_delta,
                sensitivity_score,
            });
        }
        Ok(Fp8SensitivityReport {
            schema_version: 1,
            evaluation_sequences: subset.len(),
            decode_positions_per_sequence: 4,
            sites,
        })
    }

    fn search_fp8_policy(
        &mut self,
        base_policy: &Fp8PrecisionPolicy,
        sensitivity: &Fp8SensitivityReport,
        sequences: &[Vec<u32>],
    ) -> Result<Fp8PolicySearchReport> {
        let mut selected = base_policy.clone();
        selected.name = "policy_auto_selective".to_string();
        for site in &mut selected.sites {
            site.enabled = false;
        }
        let mut ranked: Vec<_> = sensitivity.sites.iter().collect();
        ranked.sort_by(|left, right| {
            let left_score = left.expected_decode_saving_us / (left.sensitivity_score + 1.0e-6);
            let right_score = right.expected_decode_saving_us / (right.sensitivity_score + 1.0e-6);
            right_score.total_cmp(&left_score)
        });
        let subset = &sequences[..sequences.len().min(2)];
        let mut steps = Vec::with_capacity(ranked.len());
        for candidate in ranked {
            let index = selected
                .sites
                .iter()
                .position(|site| site.site == candidate.site)
                .context("ranked sensitivity site missing from policy")?;
            selected.sites[index].enabled = true;
            let quality = self.evaluate_fp8_policy_sampled(&selected, subset, 4)?;
            let final_hidden = quality
                .propagation
                .iter()
                .find(|point| point.point == "final_rms_norm")
                .context("policy search propagation missing final RMSNorm")?;
            let accepted = quality.metrics.non_finite_logits == 0
                && quality.metrics.relative_nll_delta <= 0.005
                && quality.metrics.mean_kl_bf16_to_candidate <= 0.05
                && final_hidden.nrmse <= 0.12
                && final_hidden.cosine >= 0.985;
            if !accepted {
                selected.sites[index].enabled = false;
            }
            let enabled_sites_after_step =
                selected.sites.iter().filter(|site| site.enabled).count();
            eprintln!(
                "policy search {} {} (enabled={enabled_sites_after_step})",
                candidate.site,
                if accepted { "accepted" } else { "rejected" }
            );
            steps.push(Fp8PolicySearchStep {
                site: candidate.site.clone(),
                expected_decode_saving_us: candidate.expected_decode_saving_us,
                risk_score: candidate.sensitivity_score,
                accepted,
                enabled_sites_after_step,
                relative_nll_delta: quality.metrics.relative_nll_delta,
                mean_logit_kl: quality.metrics.mean_kl_bf16_to_candidate,
                final_hidden_nrmse: final_hidden.nrmse,
                final_hidden_cosine: final_hidden.cosine,
            });
        }
        ensure!(
            selected.sites.iter().any(|site| site.enabled),
            "automatic FP8 search rejected every site"
        );
        Ok(Fp8PolicySearchReport {
            schema_version: 1,
            ranking: "expected_decode_saving_us/(single_site_sensitivity_score+1e-6)",
            fast_gate: "relative_nll_delta<=0.5%,mean_KL<=0.05,final_hidden_nrmse<=0.12,final_hidden_cosine>=0.985,no_nonfinite",
            steps,
            selected_policy: selected,
        })
    }

    fn run_fp8_greedy_diagnostics(
        &mut self,
        policy: &Fp8PrecisionPolicy,
    ) -> Result<Fp8GreedyDiagnostics> {
        let enabled = self.model.install_fp8_policy(&self.runtime, policy)?;
        ensure!(enabled > 0, "greedy FP8 policy enables no sites");
        let prompts = [
            "Who are you?",
            "Explain why the sky appears blue.",
            "Write a short Rust function that adds two integers.",
            "What is the capital of France?",
            "Summarize the purpose of a GPU in one paragraph.",
            "List three benefits of regular exercise.",
            "Continue this sequence: 1, 1, 2, 3, 5,",
            "Describe a quiet evening by the sea.",
        ];
        let options = GenerationOptions {
            max_new_tokens: 32,
            sampling: SamplingConfig {
                temperature: 0.0,
                top_k: 50,
                repetition_penalty: 1.0,
                seed: DEFAULT_SAMPLING_SEED,
            },
        };
        let mut diagnostics = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            self.model.set_decode_fp8_enabled(false)?;
            let reference = self.generate(prompt, options)?;
            self.model.set_decode_fp8_enabled(true)?;
            let candidate = self.generate(prompt, options)?;
            let common = reference.token_ids.len().min(candidate.token_ids.len());
            let first_divergent_token = reference
                .token_ids
                .iter()
                .zip(&candidate.token_ids)
                .position(|(reference, candidate)| reference != candidate)
                .or_else(|| {
                    (reference.token_ids.len() != candidate.token_ids.len()).then_some(common)
                });
            diagnostics.push(Fp8GreedyPromptDiagnostic {
                prompt: prompt.to_string(),
                bf16_text: reference.text,
                candidate_text: candidate.text,
                bf16_tokens: reference.token_ids.len(),
                candidate_tokens: candidate.token_ids.len(),
                first_divergent_token,
                agreement_before_divergence: first_divergent_token.unwrap_or(common),
                exact_sequence_agreement: reference.token_ids == candidate.token_ids,
                output_length_agreement: reference.token_ids.len() == candidate.token_ids.len(),
            });
        }
        self.model.set_decode_fp8_enabled(false)?;
        let count = diagnostics.len() as f64;
        let exact_sequence_agreement_rate = diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.exact_sequence_agreement)
            .count() as f64
            / count;
        let output_length_agreement_rate = diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.output_length_agreement)
            .count() as f64
            / count;
        Ok(Fp8GreedyDiagnostics {
            policy_name: policy.name.clone(),
            prompts: diagnostics.len(),
            exact_sequence_agreement_rate,
            output_length_agreement_rate,
            diagnostics,
        })
    }

    pub fn generate(&self, prompt: &str, options: GenerationOptions) -> Result<GenerationResult> {
        self.generate_impl(prompt, options, false)
    }

    fn generate_fixed_steps(
        &self,
        prompt: &str,
        options: GenerationOptions,
    ) -> Result<GenerationResult> {
        self.generate_impl(prompt, options, true)
    }

    fn generate_impl(
        &self,
        prompt: &str,
        options: GenerationOptions,
        ignore_eos: bool,
    ) -> Result<GenerationResult> {
        let request_started = Instant::now();
        let pool_stats_started = self.runtime.bf16_pool_stats();
        let fp8_pool_stats_started = self.runtime.fp8_pool_stats();
        ensure!(
            options.max_new_tokens > 0,
            "max_new_tokens must be positive"
        );
        let mut sampler = Sampler::new(options.sampling)?;
        let mut decode_profile = ModelProfileRecorder::new(
            self.config.decode_profile,
            self.config.decode_profile_warmup_steps,
            self.config.decode_profile_steps,
        )?;

        let tokenization_started = Instant::now();
        let prompt_ids = self.tokenizer.encode_user_prompt(prompt)?;
        let tokenization_ms = elapsed_ms(tokenization_started);
        ensure!(!prompt_ids.is_empty(), "tokenized prompt is empty");
        let capacity = prompt_ids
            .len()
            .checked_add(options.max_new_tokens)
            .ok_or_else(|| anyhow::anyhow!("requested sequence length overflow"))?;
        ensure!(
            capacity <= self.model.config().max_position_embeddings,
            "requested sequence length {capacity} exceeds model limit {}",
            self.model.config().max_position_embeddings
        );
        let cache_gpu_started = self.runtime.record_timing_event()?;
        let cache_allocation_started = Instant::now();
        let mut cache = self
            .model
            .new_cache(&self.runtime, capacity, self.config.kv_page_size)?;
        let cache_allocation_cpu_ms = elapsed_ms(cache_allocation_started);
        let cache_gpu_finished = self.runtime.record_timing_event()?;

        let prefill_gpu_started = self.runtime.record_timing_event()?;
        let mut logits = self
            .model
            .forward_logits(&self.runtime, &mut cache, &prompt_ids)?;
        let prefill_gpu_finished = self.runtime.record_timing_event()?;

        let mut generated = Vec::with_capacity(options.max_new_tokens);
        let mut history = prompt_ids.clone();
        let mut finish_reason = "length";
        let mut gpu_wait_and_sampling_total_ms = 0.0f64;
        let mut decode_gpu_ms = 0.0f64;
        let mut inter_token_ms = Vec::with_capacity(options.max_new_tokens.saturating_sub(1));

        let first_sampling_started = Instant::now();
        let mut token = sampler.sample(&self.runtime, &logits, &history)?;
        let first_token_gpu_wait_and_sampling_ms = elapsed_ms(first_sampling_started);
        gpu_wait_and_sampling_total_ms += first_token_gpu_wait_and_sampling_ms;

        let first_token_ready = Instant::now();
        let ttft_ms = first_token_ready
            .duration_since(request_started)
            .as_secs_f64()
            * 1000.0;
        let cache_initialization_gpu_ms = self
            .runtime
            .elapsed_ms(&cache_gpu_started, &cache_gpu_finished)?;
        let prefill_gpu_ms = self
            .runtime
            .elapsed_ms(&prefill_gpu_started, &prefill_gpu_finished)?;
        let mut last_visible_token_ready = first_token_ready;
        let decode_bf16_pool_started = self.runtime.bf16_pool_stats();
        let decode_fp8_pool_started = self.runtime.fp8_pool_stats();

        for step in 0..options.max_new_tokens {
            if token == self.model.config().eos_token_id && !ignore_eos {
                finish_reason = "stop";
                break;
            }
            generated.push(token);
            history.push(token);

            if step + 1 < options.max_new_tokens {
                if let Some(profile) = decode_profile.as_mut() {
                    profile.start_step(&self.runtime)?;
                }
                let decode_gpu_started = self.runtime.record_timing_event()?;
                logits = self.model.forward_logits_profiled(
                    &self.runtime,
                    &mut cache,
                    &[token],
                    decode_profile.as_mut(),
                )?;
                let decode_gpu_finished = self.runtime.record_timing_event()?;

                let sampling_started = Instant::now();
                token = match decode_profile.as_mut() {
                    Some(profile) => {
                        profile.region(&self.runtime, ProfileRegion::Sampling, || {
                            sampler.sample(&self.runtime, &logits, &history)
                        })?
                    }
                    None => sampler.sample(&self.runtime, &logits, &history)?,
                };
                if let Some(profile) = decode_profile.as_mut() {
                    profile.finish_step(&self.runtime)?;
                }
                gpu_wait_and_sampling_total_ms += elapsed_ms(sampling_started);
                decode_gpu_ms += self
                    .runtime
                    .elapsed_ms(&decode_gpu_started, &decode_gpu_finished)?;

                let token_ready = Instant::now();
                if token != self.model.config().eos_token_id {
                    inter_token_ms.push(
                        token_ready
                            .duration_since(last_visible_token_ready)
                            .as_secs_f64()
                            * 1000.0,
                    );
                    last_visible_token_ready = token_ready;
                }
            }
        }

        let decode_total_ms = last_visible_token_ready
            .duration_since(first_token_ready)
            .as_secs_f64()
            * 1000.0;
        let tpot_mean_ms = mean(&inter_token_ms);
        let tpot_p50_ms = percentile(&inter_token_ms, 0.50);
        let tpot_p95_ms = percentile(&inter_token_ms, 0.95);
        let decode_tokens_per_second = tpot_mean_ms
            .filter(|value| *value > 0.0)
            .map(|value| 1000.0 / value);

        let detokenization_started = Instant::now();
        let text = self.tokenizer.decode(&generated)?;
        let detokenization_ms = elapsed_ms(detokenization_started);
        let total_ms = elapsed_ms(request_started);
        let pool_stats_finished = self.runtime.bf16_pool_stats();
        let fp8_pool_stats_finished = self.runtime.fp8_pool_stats();
        let profile = match decode_profile {
            Some(profile) if profile.has_steps() => Some(profile.report()?),
            _ => None,
        };
        let completion_tokens = generated.len();

        Ok(GenerationResult {
            text,
            token_ids: generated,
            prompt_tokens: prompt_ids.len(),
            completion_tokens,
            finish_reason,
            profile,
            metrics: GenerationMetrics {
                tokenization_ms,
                queue_delay_ms: 0.0,
                scheduler_cpu_ms: 0.0,
                cache_allocation_cpu_ms,
                cache_initialization_gpu_ms,
                prefill_gpu_ms,
                prefill_submit_cpu_ms: 0.0,
                prefill_d2h_ms: 0.0,
                first_token_gpu_wait_and_sampling_ms,
                ttft_ms,
                decode_gpu_ms,
                decode_submit_cpu_ms: 0.0,
                decode_d2h_ms: 0.0,
                decode_total_ms,
                tpot_mean_ms,
                tpot_p50_ms,
                tpot_p95_ms,
                decode_tokens_per_second,
                gpu_wait_and_sampling_total_ms,
                bf16_pool_hits: pool_stats_finished
                    .hits
                    .saturating_sub(pool_stats_started.hits),
                bf16_pool_misses: pool_stats_finished
                    .misses
                    .saturating_sub(pool_stats_started.misses),
                fp8_pool_hits: fp8_pool_stats_finished
                    .hits
                    .saturating_sub(fp8_pool_stats_started.hits),
                fp8_pool_misses: fp8_pool_stats_finished
                    .misses
                    .saturating_sub(fp8_pool_stats_started.misses),
                decode_bf16_pool_hits: pool_stats_finished
                    .hits
                    .saturating_sub(decode_bf16_pool_started.hits),
                decode_bf16_pool_misses: pool_stats_finished
                    .misses
                    .saturating_sub(decode_bf16_pool_started.misses),
                decode_fp8_pool_hits: fp8_pool_stats_finished
                    .hits
                    .saturating_sub(decode_fp8_pool_started.hits),
                decode_fp8_pool_misses: fp8_pool_stats_finished
                    .misses
                    .saturating_sub(decode_fp8_pool_started.misses),
                bf16_pool_available_elements: pool_stats_finished.available_elements,
                bf16_pool_dropped_elements: pool_stats_finished.dropped_elements,
                fp8_pool_available_elements: fp8_pool_stats_finished.available_elements,
                fp8_pool_dropped_elements: fp8_pool_stats_finished.dropped_elements,
                bf16_pool_internal_fragment_elements: pool_stats_finished
                    .internal_fragment_elements,
                fp8_pool_internal_fragment_elements: fp8_pool_stats_finished
                    .internal_fragment_elements,
                detokenization_ms,
                total_ms,
            },
        })
    }
}

fn calibration_sequences(
    reader: impl BufRead,
    tokenizer: &Lfm2Tokenizer,
    bos_token_id: u32,
    max_sequences: usize,
    max_sequence_tokens: usize,
) -> Result<Vec<Vec<u32>>> {
    ensure!(
        max_sequence_tokens >= 64,
        "calibration sequences require at least 64 tokens"
    );
    let nominal_lengths = [96usize, 192, 384, 768];
    let mut token_stream = Vec::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!("failed to read calibration corpus line {}", line_index + 1)
        })?;
        let Some(text) = calibration_text_from_line(&line)
            .with_context(|| format!("invalid calibration corpus line {}", line_index + 1))?
        else {
            continue;
        };
        token_stream.extend(tokenizer.encode_text(&text)?);
        if token_stream.len()
            >= max_sequences
                .checked_mul(max_sequence_tokens)
                .context("calibration token target overflow")?
        {
            break;
        }
    }

    let mut sequences = Vec::with_capacity(max_sequences);
    let mut offset = 0usize;
    while sequences.len() < max_sequences {
        let bucket = sequences.len() % 20;
        let nominal = match bucket {
            0..=4 => nominal_lengths[0],
            5..=11 => nominal_lengths[1],
            12..=16 => nominal_lengths[2],
            _ => nominal_lengths[3],
        };
        let length = nominal.min(max_sequence_tokens);
        let payload = length.saturating_sub(1);
        let end = offset
            .checked_add(payload)
            .context("calibration corpus offset overflow")?;
        if end > token_stream.len() {
            break;
        }
        let mut sequence = Vec::with_capacity(length);
        sequence.push(bos_token_id);
        sequence.extend_from_slice(&token_stream[offset..end]);
        sequences.push(sequence);
        offset = end;
    }
    Ok(sequences)
}

fn decode_sample_positions(sequence_tokens: usize, requested: usize) -> Vec<usize> {
    if sequence_tokens < 2 || requested == 0 {
        return Vec::new();
    }
    let first = 1usize.min(sequence_tokens - 1);
    let span = sequence_tokens - 1 - first;
    let count = requested.min(span + 1);
    let mut positions = Vec::with_capacity(count);
    for index in 0..count {
        let position = if count == 1 {
            sequence_tokens - 1
        } else {
            first + span * index / (count - 1)
        };
        if positions.last().copied() != Some(position) {
            positions.push(position);
        }
    }
    positions
}

fn calibration_text_from_line(line: &str) -> Result<Option<String>> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    if trimmed.starts_with('{') {
        let value: serde_json::Value =
            serde_json::from_str(trimmed).context("failed to parse JSONL object")?;
        let text = value
            .get("text")
            .or_else(|| value.get("prompt"))
            .and_then(serde_json::Value::as_str)
            .context("JSONL calibration row requires string field `text` or `prompt`")?;
        return Ok(Some(text.to_string()));
    }
    if trimmed.starts_with('"')
        && let Ok(text) = serde_json::from_str::<String>(trimmed)
    {
        return Ok(Some(text));
    }

    Ok(Some(trimmed.to_string()))
}

fn load_fp8_policy(path: &Path) -> Result<Fp8PrecisionPolicy> {
    let file = File::open(path)
        .with_context(|| format!("failed to open FP8 policy {}", path.display()))?;
    serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("failed to parse FP8 policy {}", path.display()))
}

fn summarize_generation_metrics(samples: &[GenerationMetrics]) -> Result<Fp8BenchmarkSummary> {
    ensure!(!samples.is_empty(), "cannot summarize an empty benchmark");
    let ttft = samples
        .iter()
        .map(|sample| sample.ttft_ms)
        .collect::<Vec<_>>();
    let tpot = samples
        .iter()
        .map(|sample| sample.tpot_mean_ms.context("benchmark sample has no TPOT"))
        .collect::<Result<Vec<_>>>()?;
    let total = samples
        .iter()
        .map(|sample| sample.total_ms)
        .collect::<Vec<_>>();
    Ok(Fp8BenchmarkSummary {
        ttft_mean_ms: mean(&ttft).context("missing TTFT mean")?,
        ttft_p50_ms: percentile(&ttft, 0.50).context("missing TTFT p50")?,
        ttft_p95_ms: percentile(&ttft, 0.95).context("missing TTFT p95")?,
        tpot_mean_ms: mean(&tpot).context("missing TPOT mean")?,
        tpot_p50_ms: percentile(&tpot, 0.50).context("missing TPOT p50")?,
        tpot_p95_ms: percentile(&tpot, 0.95).context("missing TPOT p95")?,
        total_mean_ms: mean(&total).context("missing total mean")?,
        bf16_pool_hits: samples.iter().map(|sample| sample.bf16_pool_hits).sum(),
        bf16_pool_misses: samples.iter().map(|sample| sample.bf16_pool_misses).sum(),
        fp8_pool_hits: samples.iter().map(|sample| sample.fp8_pool_hits).sum(),
        fp8_pool_misses: samples.iter().map(|sample| sample.fp8_pool_misses).sum(),
        decode_bf16_pool_hits: samples
            .iter()
            .map(|sample| sample.decode_bf16_pool_hits)
            .sum(),
        decode_bf16_pool_misses: samples
            .iter()
            .map(|sample| sample.decode_bf16_pool_misses)
            .sum(),
        decode_fp8_pool_hits: samples
            .iter()
            .map(|sample| sample.decode_fp8_pool_hits)
            .sum(),
        decode_fp8_pool_misses: samples
            .iter()
            .map(|sample| sample.decode_fp8_pool_misses)
            .sum(),
    })
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn cpu_argmax_bf16(values: &[half::bf16]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.to_f32().total_cmp(&right.1.to_f32()))
        .map(|(index, _)| index)
}

fn mean(samples: &[f64]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    Some(samples.iter().sum::<f64>() / samples.len() as f64)
}

fn percentile(samples: &[f64], percentile: f64) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
    Some(sorted[index])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distribution_metrics_handle_empty_and_ordered_samples() {
        assert_eq!(mean(&[]), None);
        assert_eq!(percentile(&[], 0.95), None);

        let samples = [4.0, 1.0, 3.0, 2.0];
        assert_eq!(mean(&samples), Some(2.5));
        assert_eq!(percentile(&samples, 0.50), Some(3.0));
        assert_eq!(percentile(&samples, 0.95), Some(4.0));
    }

    #[test]
    fn calibration_corpus_supports_text_and_jsonl() -> Result<()> {
        assert_eq!(
            calibration_text_from_line("  hello  ")?,
            Some("hello".into())
        );
        assert_eq!(
            calibration_text_from_line(r#"{"text":"world"}"#)?,
            Some("world".into())
        );
        assert_eq!(calibration_text_from_line("  ")?, None);
        assert_eq!(
            calibration_text_from_line(r#""ordinary quoted prose without a closing JSON quote"#)?,
            Some(r#""ordinary quoted prose without a closing JSON quote"#.into())
        );
        assert!(calibration_text_from_line(r#"{"id":1}"#).is_err());
        Ok(())
    }
}
