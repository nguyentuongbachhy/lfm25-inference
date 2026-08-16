use std::collections::{BTreeMap, HashMap};

use anyhow::{Context as _, Result, ensure};
use half::bf16;
use serde::{Deserialize, Serialize};

use crate::{cuda::CudaRuntime, tensor::Tensor};

use super::quantization::{
    ScaleStrategy, TensorwideE4m3ScaleCandidates, tensorwide_e4m3_scale_candidates,
};

const BF16_MAGNITUDE_BINS: usize = 1 << 15;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationPhase {
    Weight,
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationTensorKind {
    Activation,
    Weight,
}

struct TensorStatisticsAccumulator {
    kind: CalibrationTensorKind,
    feature_size: usize,
    min_rows: usize,
    max_rows: usize,
    observations: u64,
    finite_values: u64,
    non_finite_values: u64,
    zeros: u64,
    sum_abs: f64,
    sum_squares: f64,
    amax: f32,
    magnitude_histogram: Vec<u64>,
}

impl TensorStatisticsAccumulator {
    fn new(kind: CalibrationTensorKind, feature_size: usize, rows: usize) -> Self {
        Self {
            kind,
            feature_size,
            min_rows: rows,
            max_rows: rows,
            observations: 0,
            finite_values: 0,
            non_finite_values: 0,
            zeros: 0,
            sum_abs: 0.0,
            sum_squares: 0.0,
            amax: 0.0,
            magnitude_histogram: vec![0; BF16_MAGNITUDE_BINS],
        }
    }

    fn observe(&mut self, values: &[bf16], feature_size: usize, rows: usize) -> Result<()> {
        ensure!(
            self.feature_size == feature_size,
            "calibration feature size changed from {} to {feature_size}",
            self.feature_size
        );
        self.min_rows = self.min_rows.min(rows);
        self.max_rows = self.max_rows.max(rows);
        self.observations = self.observations.saturating_add(1);

        for value in values {
            let magnitude_bits = usize::from(value.to_bits() & 0x7fff);
            let magnitude = bf16::from_bits(magnitude_bits as u16).to_f32();

            if !magnitude.is_finite() {
                self.non_finite_values = self.non_finite_values.saturating_add(1);
                continue;
            }

            self.finite_values = self.finite_values.saturating_add(1);
            self.zeros = self.zeros.saturating_add(u64::from(magnitude == 0.0));
            self.sum_abs += f64::from(magnitude);
            self.sum_squares += f64::from(magnitude) * f64::from(magnitude);
            self.amax = self.amax.max(magnitude);
            self.magnitude_histogram[magnitude_bits] =
                self.magnitude_histogram[magnitude_bits].saturating_add(1);
        }

        Ok(())
    }

    fn percentile(&self, quantile: f64) -> f32 {
        if self.finite_values == 0 {
            return 0.0;
        }
        let rank = (quantile * self.finite_values as f64).ceil() as u64;
        let target = rank.max(1);
        let mut cumulative = 0_u64;

        for (bits, count) in self.magnitude_histogram.iter().enumerate() {
            cumulative = cumulative.saturating_add(*count);
            if cumulative >= target {
                return bf16::from_bits(bits as u16).to_f32();
            }
        }

        self.amax
    }

    fn fraction_above(&self, threshold: f32) -> f64 {
        if self.finite_values == 0 || !threshold.is_finite() || threshold < 0.0 {
            return 0.0;
        }
        let threshold_bin = usize::from(bf16::from_f32(threshold).to_bits() & 0x7fff);
        let above = self.magnitude_histogram[threshold_bin.saturating_add(1)..]
            .iter()
            .fold(0_u64, |total, count| total.saturating_add(*count));
        above as f64 / self.finite_values as f64
    }

    fn finish(self, phase: CalibrationPhase, name: String) -> CalibrationTensorReport {
        let p99 = self.percentile(0.99);
        let p99_9 = self.percentile(0.999);
        let p99_99 = self.percentile(0.9999);
        let mean_abs = if self.finite_values == 0 {
            0.0
        } else {
            (self.sum_abs / self.finite_values as f64) as f32
        };
        let rms = if self.finite_values == 0 {
            0.0
        } else {
            (self.sum_squares / self.finite_values as f64).sqrt() as f32
        };
        let amax_over_rms = positive_ratio(self.amax, rms);
        let amax_over_p99_99 = positive_ratio(self.amax, p99_99);
        let fraction_above_p99 = self.fraction_above(p99);
        let fraction_above_p99_9 = self.fraction_above(p99_9);
        let fraction_above_p99_99 = self.fraction_above(p99_99);

        CalibrationTensorReport {
            name,
            phase,
            kind: self.kind,
            feature_size: self.feature_size,
            min_rows: self.min_rows,
            max_rows: self.max_rows,
            observations: self.observations,
            finite_values: self.finite_values,
            non_finite_values: self.non_finite_values,
            zeros: self.zeros,
            amax: self.amax,
            mean_abs,
            rms,
            p99,
            p99_9,
            p99_99,
            amax_over_rms,
            amax_over_p99_99,
            fraction_above_p99,
            fraction_above_p99_9,
            fraction_above_p99_99,
            tensorwide_e4m3: tensorwide_e4m3_scale_candidates(self.amax, p99, p99_9, p99_99),
        }
    }
}

fn positive_ratio(numerator: f32, denominator: f32) -> Option<f32> {
    if numerator.is_finite() && denominator.is_finite() && denominator > 0.0 {
        Some(numerator / denominator)
    } else {
        None
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CalibrationTensorReport {
    pub name: String,
    pub phase: CalibrationPhase,
    pub kind: CalibrationTensorKind,
    pub feature_size: usize,
    pub min_rows: usize,
    pub max_rows: usize,
    pub observations: u64,
    pub finite_values: u64,
    pub non_finite_values: u64,
    pub zeros: u64,
    pub amax: f32,
    pub mean_abs: f32,
    pub rms: f32,
    pub p99: f32,
    pub p99_9: f32,
    pub p99_99: f32,
    pub amax_over_rms: Option<f32>,
    pub amax_over_p99_99: Option<f32>,
    pub fraction_above_p99: f64,
    pub fraction_above_p99_9: f64,
    pub fraction_above_p99_99: f64,
    pub tensorwide_e4m3: TensorwideE4m3ScaleCandidates,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Fp8CalibrationReport {
    pub schema_version: u32,
    pub model_family: &'static str,
    pub checkpoint_path: String,
    pub corpus_path: String,
    pub requested_max_sequences: usize,
    pub requested_max_sequence_tokens: usize,
    pub kv_page_size: usize,
    pub scale_policy_status: &'static str,
    pub weight_representation: &'static str,
    pub activation_sampling: &'static str,
    pub percentile_method: &'static str,
    pub sequences: usize,
    pub tokens: usize,
    pub min_sequence_tokens: usize,
    pub max_sequence_tokens: usize,
    pub prefill_forward_calls: usize,
    pub decode_forward_calls: usize,
    pub min_decode_context: usize,
    pub max_decode_context: usize,
    pub tensors: Vec<CalibrationTensorReport>,
}

impl Fp8CalibrationReport {
    pub fn activation_outliers(&self, limit: usize) -> Vec<&CalibrationTensorReport> {
        let mut tensors = self
            .tensors
            .iter()
            .filter(|tensor| tensor.kind == CalibrationTensorKind::Activation)
            .collect::<Vec<_>>();
        tensors.sort_by(|left, right| {
            right
                .amax_over_p99_99
                .unwrap_or(0.0)
                .total_cmp(&left.amax_over_p99_99.unwrap_or(0.0))
        });
        tensors.truncate(limit.min(tensors.len()));
        tensors
    }
}

struct ActivationSamples {
    feature_size: usize,
    rows: usize,
    seen: u64,
    values: Vec<bf16>,
}

pub(crate) struct CalibrationCollector {
    tensors: HashMap<(CalibrationPhase, String), TensorStatisticsAccumulator>,
    activation_phase: CalibrationPhase,
    decode_sample_limit: usize,
    decode_samples: HashMap<String, ActivationSamples>,
    sequences: usize,
    tokens: usize,
    min_sequence_tokens: usize,
    max_sequence_tokens: usize,
    prefill_forward_calls: usize,
    decode_forward_calls: usize,
    min_decode_context: usize,
    max_decode_context: usize,
}

impl CalibrationCollector {
    pub(crate) fn new(decode_sample_limit: usize) -> Self {
        Self {
            tensors: HashMap::new(),
            activation_phase: CalibrationPhase::Prefill,
            decode_sample_limit,
            decode_samples: HashMap::new(),
            sequences: 0,
            tokens: 0,
            min_sequence_tokens: usize::MAX,
            max_sequence_tokens: 0,
            prefill_forward_calls: 0,
            decode_forward_calls: 0,
            min_decode_context: usize::MAX,
            max_decode_context: 0,
        }
    }

    pub(crate) fn set_activation_phase(&mut self, phase: CalibrationPhase) -> Result<()> {
        ensure!(
            phase != CalibrationPhase::Weight,
            "activation calibration cannot use the weight phase"
        );
        self.activation_phase = phase;
        Ok(())
    }

    pub(crate) fn observe(
        &mut self,
        runtime: &CudaRuntime,
        name: impl Into<String>,
        kind: CalibrationTensorKind,
        tensor: &Tensor<bf16>,
    ) -> Result<()> {
        ensure!(tensor.rank() > 0, "cannot calibrate a rank-zero tensor");
        let feature_size = tensor.dims()[tensor.rank() - 1];
        ensure!(
            feature_size > 0,
            "cannot calibrate an empty feature dimension"
        );
        let rows = tensor.numel() / feature_size;
        let values = runtime
            .download(tensor)
            .context("failed to read back tensor for FP8 calibration")?;
        let name = name.into();
        let phase = if kind == CalibrationTensorKind::Weight {
            CalibrationPhase::Weight
        } else {
            self.activation_phase
        };
        self.maybe_store_decode_sample(&name, feature_size, rows, &values)?;
        let accumulator = self
            .tensors
            .entry((phase, name))
            .or_insert_with(|| TensorStatisticsAccumulator::new(kind, feature_size, rows));
        ensure!(
            accumulator.kind == kind,
            "calibration tensor kind changed across observations"
        );
        accumulator.observe(&values, feature_size, rows)
    }

    pub(crate) fn observe_last_row(
        &mut self,
        runtime: &CudaRuntime,
        name: impl Into<String>,
        tensor: &Tensor<bf16>,
    ) -> Result<()> {
        ensure!(tensor.rank() == 2, "last-row calibration requires rank 2");
        let feature_size = tensor.dims()[1];
        let rows = tensor.dims()[0];
        ensure!(
            rows > 0 && feature_size > 0,
            "last-row calibration requires data"
        );
        let start = (rows - 1)
            .checked_mul(feature_size)
            .context("last-row calibration offset overflow")?;
        let end = start
            .checked_add(feature_size)
            .context("last-row calibration end overflow")?;
        let last_row = tensor
            .storage()
            .try_slice(start..end)
            .context("invalid last-row calibration range")?;
        let values = runtime
            .stream()
            .clone_dtoh(&last_row)
            .context("failed to read back last-row calibration tensor")?;
        let name = name.into();
        self.maybe_store_decode_sample(&name, feature_size, 1, &values)?;
        let accumulator = self
            .tensors
            .entry((self.activation_phase, name))
            .or_insert_with(|| {
                TensorStatisticsAccumulator::new(CalibrationTensorKind::Activation, feature_size, 1)
            });
        accumulator.observe(&values, feature_size, 1)
    }

    fn maybe_store_decode_sample(
        &mut self,
        name: &str,
        feature_size: usize,
        rows: usize,
        values: &[bf16],
    ) -> Result<()> {
        if self.activation_phase != CalibrationPhase::Decode
            || self.decode_sample_limit == 0
            || rows != 1
        {
            return Ok(());
        }
        let samples = self
            .decode_samples
            .entry(name.to_string())
            .or_insert_with(|| ActivationSamples {
                feature_size,
                rows: 0,
                seen: 0,
                values: Vec::with_capacity(feature_size.saturating_mul(self.decode_sample_limit)),
            });
        ensure!(
            samples.feature_size == feature_size,
            "decode sample feature size changed for {name}"
        );
        samples.seen = samples.seen.saturating_add(1);
        if samples.rows < self.decode_sample_limit {
            samples.values.extend_from_slice(values);
            samples.rows += 1;
        } else {
            let mut random = samples.seen.wrapping_add(0x9e37_79b9_7f4a_7c15);
            random = (random ^ (random >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            random = (random ^ (random >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            random ^= random >> 31;
            let selected = usize::try_from(random % samples.seen)
                .context("decode reservoir index exceeds usize")?;
            if selected < self.decode_sample_limit {
                let start = selected
                    .checked_mul(feature_size)
                    .context("decode reservoir offset overflow")?;
                let end = start
                    .checked_add(feature_size)
                    .context("decode reservoir end overflow")?;
                samples.values[start..end].copy_from_slice(values);
            }
        }
        Ok(())
    }

    pub(crate) fn record_sequence(&mut self, tokens: usize) -> Result<()> {
        ensure!(tokens > 0, "calibration sequence must contain tokens");
        self.sequences = self.sequences.saturating_add(1);
        self.tokens = self
            .tokens
            .checked_add(tokens)
            .context("calibration token count overflow")?;
        self.min_sequence_tokens = self.min_sequence_tokens.min(tokens);
        self.max_sequence_tokens = self.max_sequence_tokens.max(tokens);
        Ok(())
    }

    pub(crate) fn sequence_count(&self) -> usize {
        self.sequences
    }

    pub(crate) fn record_prefill_forward(&mut self) {
        self.prefill_forward_calls = self.prefill_forward_calls.saturating_add(1);
    }

    pub(crate) fn record_decode_forward(&mut self, context: usize) -> Result<()> {
        ensure!(context > 0, "decode calibration context must be positive");
        self.decode_forward_calls = self.decode_forward_calls.saturating_add(1);
        self.min_decode_context = self.min_decode_context.min(context);
        self.max_decode_context = self.max_decode_context.max(context);
        Ok(())
    }

    pub(crate) fn decode_samples(&self, name: &str) -> Option<(&[bf16], usize, usize)> {
        self.decode_samples.get(name).map(|samples| {
            (
                samples.values.as_slice(),
                samples.rows,
                samples.feature_size,
            )
        })
    }

    pub(crate) fn scale_candidates(
        &self,
        phase: CalibrationPhase,
        name: &str,
    ) -> Option<TensorwideE4m3ScaleCandidates> {
        let accumulator = self.tensors.get(&(phase, name.to_string()))?;
        Some(tensorwide_e4m3_scale_candidates(
            accumulator.amax,
            accumulator.percentile(0.99),
            accumulator.percentile(0.999),
            accumulator.percentile(0.9999),
        ))
    }

    pub(crate) fn clipping_at(
        &self,
        phase: CalibrationPhase,
        name: &str,
        strategy: ScaleStrategy,
    ) -> Option<(u64, f64)> {
        let accumulator = self.tensors.get(&(phase, name.to_string()))?;
        let threshold = self.scale_candidates(phase, name)?.get(strategy).clip_amax;
        let fraction = accumulator.fraction_above(threshold);
        let count = (fraction * accumulator.finite_values as f64).round() as u64;
        Some((count, fraction))
    }

    pub(crate) fn finish(
        self,
        checkpoint_path: String,
        corpus_path: String,
        requested_max_sequences: usize,
        requested_max_sequence_tokens: usize,
        kv_page_size: usize,
    ) -> Result<Fp8CalibrationReport> {
        ensure!(
            self.sequences > 0,
            "calibration corpus produced no sequences"
        );
        ensure!(
            self.prefill_forward_calls == self.sequences,
            "expected one calibrated prefill per sequence, got {} for {} sequences",
            self.prefill_forward_calls,
            self.sequences
        );
        ensure!(
            self.decode_forward_calls > 0,
            "calibration produced no decode M=1 observations"
        );
        for (phase, expected) in [
            (CalibrationPhase::Weight, 77usize),
            (CalibrationPhase::Prefill, 65usize),
            (CalibrationPhase::Decode, 65usize),
        ] {
            let actual = self
                .tensors
                .keys()
                .filter(|(tensor_phase, _)| *tensor_phase == phase)
                .count();
            ensure!(
                actual == expected,
                "expected {expected} {phase:?} calibration sites, observed {actual}"
            );
        }
        let ordered: BTreeMap<_, _> = self.tensors.into_iter().collect();
        let tensors = ordered
            .into_iter()
            .map(|((phase, name), accumulator)| accumulator.finish(phase, name))
            .collect();

        Ok(Fp8CalibrationReport {
            schema_version: 1,
            model_family: "LFM2.5",
            checkpoint_path,
            corpus_path,
            requested_max_sequences,
            requested_max_sequence_tokens,
            kv_page_size,
            scale_policy_status: "experimental_candidates_not_for_production_dispatch",
            weight_representation: "runtime_bf16_with_packed_gate_up_and_tied_lm_head",
            activation_sampling: "all_prefill_values_and_sampled_teacher_forced_decode_m1_values_at_exact_gemm_inputs",
            percentile_method: "exact_bf16_magnitude_histogram",
            sequences: self.sequences,
            tokens: self.tokens,
            min_sequence_tokens: self.min_sequence_tokens,
            max_sequence_tokens: self.max_sequence_tokens,
            prefill_forward_calls: self.prefill_forward_calls,
            decode_forward_calls: self.decode_forward_calls,
            min_decode_context: self.min_decode_context,
            max_decode_context: self.max_decode_context,
            tensors,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_histogram_percentiles_are_exact() -> Result<()> {
        let values = [1.0, -2.0, 3.0, 4.0].map(bf16::from_f32);
        let mut accumulator =
            TensorStatisticsAccumulator::new(CalibrationTensorKind::Activation, 2, 2);
        accumulator.observe(&values, 2, 2)?;
        assert_eq!(accumulator.percentile(0.50), 2.0);
        assert_eq!(accumulator.percentile(0.99), 4.0);
        assert_eq!(accumulator.amax, 4.0);
        assert_eq!(accumulator.fraction_above(2.0), 0.5);
        Ok(())
    }
}
