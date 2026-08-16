use std::collections::BTreeMap;

use anyhow::{Context as _, Result, ensure};
use half::bf16;
use serde::{Deserialize, Serialize};

use crate::{cuda::CudaRuntime, tensor::Tensor};

#[derive(Default)]
pub(crate) struct HiddenCapture {
    points: BTreeMap<String, Vec<bf16>>,
}

impl HiddenCapture {
    pub(crate) fn observe_last_row(
        &mut self,
        runtime: &CudaRuntime,
        name: impl Into<String>,
        tensor: &Tensor<bf16>,
    ) -> Result<()> {
        ensure!(tensor.rank() == 2, "hidden capture requires rank-2 tensor");
        let rows = tensor.dims()[0];
        let width = tensor.dims()[1];
        ensure!(
            rows > 0 && width > 0,
            "hidden capture requires non-empty tensor"
        );
        let start = (rows - 1)
            .checked_mul(width)
            .context("hidden capture offset overflow")?;
        let end = start
            .checked_add(width)
            .context("hidden capture end overflow")?;
        let view = tensor
            .storage()
            .try_slice(start..end)
            .context("hidden capture range is invalid")?;
        let values = runtime
            .stream()
            .clone_dtoh(&view)
            .context("failed to read hidden capture")?;
        ensure!(
            self.points.insert(name.into(), values).is_none(),
            "duplicate hidden capture point"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropagationPointMetrics {
    pub point: String,
    pub observations: u64,
    pub values: u64,
    pub cosine: f64,
    pub nrmse: f64,
    pub rms_ratio: f64,
    pub max_abs_error: f32,
    pub non_finite_values: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogitDistributionMetrics {
    pub observations: u64,
    pub reference_mean_nll: f64,
    pub candidate_mean_nll: f64,
    pub absolute_nll_delta: f64,
    pub relative_nll_delta: f64,
    pub reference_perplexity: f64,
    pub candidate_perplexity: f64,
    pub perplexity_delta: f64,
    pub mean_kl_bf16_to_candidate: f64,
    pub p50_kl_bf16_to_candidate: f64,
    pub p95_kl_bf16_to_candidate: f64,
    pub p99_kl_bf16_to_candidate: f64,
    pub mean_logit_cosine: f64,
    pub top1_agreement: f64,
    pub mean_top5_overlap: f64,
    pub mean_top10_overlap: f64,
    pub non_finite_logits: u64,
}

#[derive(Default)]
pub(crate) struct LogitMetricAccumulator {
    observations: u64,
    reference_nll: f64,
    candidate_nll: f64,
    kl_samples: Vec<f64>,
    logit_cosine: f64,
    top1_agreements: u64,
    top5_overlap: f64,
    top10_overlap: f64,
    non_finite_logits: u64,
}

impl LogitMetricAccumulator {
    pub(crate) fn observe(
        &mut self,
        reference: &[bf16],
        candidate: &[bf16],
        target: u32,
    ) -> Result<()> {
        ensure!(
            !reference.is_empty() && reference.len() == candidate.len(),
            "logit metric shape mismatch"
        );
        let target = usize::try_from(target).context("target token exceeds usize")?;
        ensure!(target < reference.len(), "target token exceeds vocabulary");
        let non_finite = reference
            .iter()
            .zip(candidate)
            .filter(|(reference, candidate)| {
                !reference.to_f32().is_finite() || !candidate.to_f32().is_finite()
            })
            .count();
        if non_finite > 0 {
            self.non_finite_logits = self.non_finite_logits.saturating_add(non_finite as u64);
            return Ok(());
        }
        let reference_max = reference
            .iter()
            .map(|value| value.to_f32())
            .max_by(f32::total_cmp)
            .context("empty reference logits")?;
        let candidate_max = candidate
            .iter()
            .map(|value| value.to_f32())
            .max_by(f32::total_cmp)
            .context("empty candidate logits")?;
        if !reference_max.is_finite() || !candidate_max.is_finite() {
            self.non_finite_logits = self
                .non_finite_logits
                .saturating_add(reference.len() as u64);
            return Ok(());
        }

        let mut reference_exp_sum = 0.0f64;
        let mut candidate_exp_sum = 0.0f64;
        let mut dot = 0.0f64;
        let mut reference_square = 0.0f64;
        let mut candidate_square = 0.0f64;
        for (reference, candidate) in reference.iter().zip(candidate) {
            let reference = reference.to_f32();
            let candidate = candidate.to_f32();
            reference_exp_sum += f64::from((reference - reference_max).exp());
            candidate_exp_sum += f64::from((candidate - candidate_max).exp());
            let reference = f64::from(reference);
            let candidate = f64::from(candidate);
            dot += reference * candidate;
            reference_square += reference * reference;
            candidate_square += candidate * candidate;
        }
        ensure!(
            reference_exp_sum > 0.0 && candidate_exp_sum > 0.0,
            "invalid logit normalizer"
        );
        let reference_log_z = f64::from(reference_max) + reference_exp_sum.ln();
        let candidate_log_z = f64::from(candidate_max) + candidate_exp_sum.ln();
        self.reference_nll += reference_log_z - f64::from(reference[target].to_f32());
        self.candidate_nll += candidate_log_z - f64::from(candidate[target].to_f32());

        let mut kl = 0.0f64;
        for (reference, candidate) in reference.iter().zip(candidate) {
            let reference_log_probability = f64::from(reference.to_f32()) - reference_log_z;
            let candidate_log_probability = f64::from(candidate.to_f32()) - candidate_log_z;
            let probability = reference_log_probability.exp();
            kl += probability * (reference_log_probability - candidate_log_probability);
        }
        self.kl_samples.push(kl.max(0.0));
        let cosine_denominator = (reference_square * candidate_square).sqrt();
        self.logit_cosine += if cosine_denominator > 0.0 {
            dot / cosine_denominator
        } else {
            1.0
        };

        let reference_top = top_indices::<10>(reference);
        let candidate_top = top_indices::<10>(candidate);
        self.top1_agreements = self
            .top1_agreements
            .saturating_add(u64::from(reference_top[0] == candidate_top[0]));
        self.top5_overlap += overlap(&reference_top[..5], &candidate_top[..5]) as f64 / 5.0;
        self.top10_overlap += overlap(&reference_top, &candidate_top) as f64 / 10.0;
        self.observations = self.observations.saturating_add(1);
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<LogitDistributionMetrics> {
        ensure!(self.observations > 0, "no logit observations");
        self.kl_samples.sort_by(f64::total_cmp);
        let count = self.observations as f64;
        let reference_mean_nll = self.reference_nll / count;
        let candidate_mean_nll = self.candidate_nll / count;
        let absolute_nll_delta = candidate_mean_nll - reference_mean_nll;
        let relative_nll_delta = absolute_nll_delta / reference_mean_nll.abs().max(1.0e-12);
        let reference_perplexity = reference_mean_nll.exp();
        let candidate_perplexity = candidate_mean_nll.exp();
        Ok(LogitDistributionMetrics {
            observations: self.observations,
            reference_mean_nll,
            candidate_mean_nll,
            absolute_nll_delta,
            relative_nll_delta,
            reference_perplexity,
            candidate_perplexity,
            perplexity_delta: candidate_perplexity - reference_perplexity,
            mean_kl_bf16_to_candidate: self.kl_samples.iter().sum::<f64>() / count,
            p50_kl_bf16_to_candidate: percentile_sorted(&self.kl_samples, 0.50),
            p95_kl_bf16_to_candidate: percentile_sorted(&self.kl_samples, 0.95),
            p99_kl_bf16_to_candidate: percentile_sorted(&self.kl_samples, 0.99),
            mean_logit_cosine: self.logit_cosine / count,
            top1_agreement: self.top1_agreements as f64 / count,
            mean_top5_overlap: self.top5_overlap / count,
            mean_top10_overlap: self.top10_overlap / count,
            non_finite_logits: self.non_finite_logits,
        })
    }
}

fn top_indices<const K: usize>(values: &[bf16]) -> [usize; K] {
    let mut indices = [usize::MAX; K];
    let mut scores = [f32::NEG_INFINITY; K];
    for (index, value) in values.iter().enumerate() {
        let score = value.to_f32();
        if score <= scores[K - 1] {
            continue;
        }
        let mut position = K - 1;
        while position > 0 && score > scores[position - 1] {
            scores[position] = scores[position - 1];
            indices[position] = indices[position - 1];
            position -= 1;
        }
        scores[position] = score;
        indices[position] = index;
    }
    indices
}

fn overlap(left: &[usize], right: &[usize]) -> usize {
    left.iter().filter(|value| right.contains(value)).count()
}

fn percentile_sorted(samples: &[f64], quantile: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let index = ((samples.len() - 1) as f64 * quantile).round() as usize;
    samples[index]
}

#[derive(Default)]
pub(crate) struct PropagationAccumulator {
    points: BTreeMap<String, VectorMetricAccumulator>,
}

impl PropagationAccumulator {
    pub(crate) fn observe(
        &mut self,
        reference: &HiddenCapture,
        candidate: &HiddenCapture,
    ) -> Result<()> {
        ensure!(
            reference.points.len() == candidate.points.len(),
            "hidden capture point count mismatch"
        );
        for (name, reference_values) in &reference.points {
            let candidate_values = candidate
                .points
                .get(name)
                .with_context(|| format!("candidate missing hidden point {name}"))?;
            self.points
                .entry(name.clone())
                .or_default()
                .observe(reference_values, candidate_values)?;
        }
        Ok(())
    }

    pub(crate) fn finish(self) -> Vec<PropagationPointMetrics> {
        self.points
            .into_iter()
            .map(|(point, accumulator)| accumulator.finish(point))
            .collect()
    }
}

#[derive(Default)]
struct VectorMetricAccumulator {
    observations: u64,
    values: u64,
    squared_reference: f64,
    squared_candidate: f64,
    squared_error: f64,
    dot: f64,
    max_abs_error: f32,
    non_finite_values: u64,
}

impl VectorMetricAccumulator {
    fn observe(&mut self, reference: &[bf16], candidate: &[bf16]) -> Result<()> {
        ensure!(
            !reference.is_empty() && reference.len() == candidate.len(),
            "hidden vector shape mismatch"
        );
        self.observations = self.observations.saturating_add(1);
        self.values = self.values.saturating_add(reference.len() as u64);
        for (reference, candidate) in reference.iter().zip(candidate) {
            let reference = reference.to_f32();
            let candidate = candidate.to_f32();
            if !reference.is_finite() || !candidate.is_finite() {
                self.non_finite_values = self.non_finite_values.saturating_add(1);
                continue;
            }
            let error = candidate - reference;
            self.max_abs_error = self.max_abs_error.max(error.abs());
            let reference = f64::from(reference);
            let candidate = f64::from(candidate);
            let error = f64::from(error);
            self.squared_reference += reference * reference;
            self.squared_candidate += candidate * candidate;
            self.squared_error += error * error;
            self.dot += reference * candidate;
        }
        Ok(())
    }

    fn finish(self, point: String) -> PropagationPointMetrics {
        let denominator = (self.squared_reference * self.squared_candidate).sqrt();
        PropagationPointMetrics {
            point,
            observations: self.observations,
            values: self.values,
            cosine: if denominator > 0.0 {
                self.dot / denominator
            } else {
                1.0
            },
            nrmse: if self.squared_reference > 0.0 {
                (self.squared_error / self.squared_reference).sqrt()
            } else {
                0.0
            },
            rms_ratio: if self.squared_reference > 0.0 {
                (self.squared_candidate / self.squared_reference).sqrt()
            } else {
                1.0
            },
            max_abs_error: self.max_abs_error,
            non_finite_values: self.non_finite_values,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_metric_identity() -> Result<()> {
        let values = [1.0, -2.0, 4.0].map(bf16::from_f32);
        let mut accumulator = VectorMetricAccumulator::default();
        accumulator.observe(&values, &values)?;
        let metrics = accumulator.finish("identity".into());
        assert_eq!(metrics.nrmse, 0.0);
        assert!((metrics.cosine - 1.0).abs() < 1.0e-12);
        Ok(())
    }
}
