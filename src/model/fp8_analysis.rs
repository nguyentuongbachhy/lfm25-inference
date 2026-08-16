use anyhow::{Context as _, Result, ensure};
use half::bf16;
use serde::{Deserialize, Serialize};

use crate::{
    cuda::CudaRuntime,
    ops,
    tensor::{Shape, Tensor},
};

use super::{
    CalibrationCollector, CalibrationPhase,
    quantization::{ScalarScale, ScaleStrategy},
};

const SCALE_STRATEGIES: [ScaleStrategy; 3] = [
    ScaleStrategy::Amax,
    ScaleStrategy::P99_99,
    ScaleStrategy::P99_9,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalPrecisionRecommendation {
    Bf16,
    Fp8Tensorwide,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemmErrorMetrics {
    pub values: usize,
    pub nrmse: f64,
    pub cosine: f64,
    pub max_abs_error: f32,
    pub max_relative_error: f32,
    pub rms_error: f64,
    pub output_rms_ratio: f64,
    pub non_finite_values: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemmScaleTrial {
    pub activation_strategy: ScaleStrategy,
    pub weight_strategy: ScaleStrategy,
    pub activation_scale: ScalarScale,
    pub weight_scale: ScalarScale,
    pub activation_clipping_count: u64,
    pub activation_clipping_fraction: f64,
    pub weight_clipping_count: u64,
    pub weight_clipping_fraction: f64,
    pub metrics: GemmErrorMetrics,
    pub passes_local_screen: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemmSiteErrorReport {
    pub site: String,
    pub activation_site: String,
    pub weight_site: String,
    pub samples: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub trials: Vec<GemmScaleTrial>,
    pub selected_trial: Option<usize>,
    pub recommendation: LocalPrecisionRecommendation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fp8GemmErrorReport {
    pub schema_version: u32,
    pub input_source: &'static str,
    pub accumulation: &'static str,
    pub local_screen: &'static str,
    pub sites: Vec<GemmSiteErrorReport>,
}

impl Fp8GemmErrorReport {
    pub(crate) fn new(sites: Vec<GemmSiteErrorReport>) -> Self {
        Self {
            schema_version: 1,
            input_source: "bounded_real_teacher_forced_decode_m1_activations",
            accumulation: "cuBLASLt_FP32_compute_BF16_output",
            local_screen: "cosine>=0.995_and_nrmse<=0.10_and_all_outputs_finite",
            sites,
        }
    }
}

pub(crate) fn characterize_gemm_site(
    runtime: &CudaRuntime,
    collector: &CalibrationCollector,
    site: String,
    activation_site: &str,
    weight_site: &str,
    weight: &Tensor<bf16>,
) -> Result<GemmSiteErrorReport> {
    let (samples, rows, feature_size) = collector
        .decode_samples(activation_site)
        .with_context(|| format!("missing decode samples for {activation_site}"))?;
    ensure!(rows > 0, "decode sample set is empty for {activation_site}");
    ensure!(
        weight.rank() == 2 && weight.dims()[1] == feature_size,
        "GEMM characterization shape mismatch for {site}"
    );
    let n = weight.dims()[0];
    let input = runtime.upload(samples, Shape::new([rows, feature_size]))?;
    let reference = ops::linear_bf16(runtime, &input, weight)?;
    let reference_host = runtime
        .download(&reference)
        .with_context(|| format!("failed to read BF16 reference for {site}"))?;
    let activation_candidates = collector
        .scale_candidates(CalibrationPhase::Decode, activation_site)
        .with_context(|| format!("missing decode scales for {activation_site}"))?;
    let weight_candidates = collector
        .scale_candidates(CalibrationPhase::Weight, weight_site)
        .with_context(|| format!("missing weight scales for {weight_site}"))?;

    let mut trials = Vec::with_capacity(SCALE_STRATEGIES.len() * SCALE_STRATEGIES.len());
    for weight_strategy in SCALE_STRATEGIES {
        let weight_scale = weight_candidates.get(weight_strategy);
        let quantized_weight = ops::quantize_weight_e4m3(runtime, weight, weight_scale)
            .with_context(|| format!("failed to quantize {weight_site}"))?;
        let (weight_clipping_count, weight_clipping_fraction) = collector
            .clipping_at(CalibrationPhase::Weight, weight_site, weight_strategy)
            .context("missing weight clipping statistics")?;

        for activation_strategy in SCALE_STRATEGIES {
            let activation_scale = activation_candidates.get(activation_strategy);
            let candidate = ops::linear_fp8_e4m3(
                runtime,
                &input,
                &quantized_weight,
                activation_scale,
                weight_scale,
            )
            .with_context(|| format!("FP8 GEMM failed for {site}"))?;
            let candidate_host = runtime
                .download(&candidate)
                .with_context(|| format!("failed to read FP8 output for {site}"))?;
            let metrics = gemm_error_metrics(&reference_host, &candidate_host)?;
            let (activation_clipping_count, activation_clipping_fraction) = collector
                .clipping_at(
                    CalibrationPhase::Decode,
                    activation_site,
                    activation_strategy,
                )
                .context("missing activation clipping statistics")?;
            let passes_local_screen =
                metrics.non_finite_values == 0 && metrics.cosine >= 0.995 && metrics.nrmse <= 0.10;
            trials.push(GemmScaleTrial {
                activation_strategy,
                weight_strategy,
                activation_scale,
                weight_scale,
                activation_clipping_count,
                activation_clipping_fraction,
                weight_clipping_count,
                weight_clipping_fraction,
                metrics,
                passes_local_screen,
            });
        }
    }

    let selected_trial = trials
        .iter()
        .enumerate()
        .filter(|(_, trial)| trial.passes_local_screen)
        .min_by(|(_, left), (_, right)| {
            left.metrics
                .nrmse
                .total_cmp(&right.metrics.nrmse)
                .then_with(|| right.metrics.cosine.total_cmp(&left.metrics.cosine))
        })
        .map(|(index, _)| index);
    let recommendation = if selected_trial.is_some() {
        LocalPrecisionRecommendation::Fp8Tensorwide
    } else {
        LocalPrecisionRecommendation::Bf16
    };

    Ok(GemmSiteErrorReport {
        site,
        activation_site: activation_site.to_string(),
        weight_site: weight_site.to_string(),
        samples: rows,
        m: rows,
        n,
        k: feature_size,
        trials,
        selected_trial,
        recommendation,
    })
}

fn gemm_error_metrics(reference: &[bf16], candidate: &[bf16]) -> Result<GemmErrorMetrics> {
    ensure!(
        !reference.is_empty() && reference.len() == candidate.len(),
        "GEMM error metric shape mismatch"
    );
    let mut squared_error = 0.0f64;
    let mut squared_reference = 0.0f64;
    let mut squared_candidate = 0.0f64;
    let mut dot = 0.0f64;
    let mut max_abs_error = 0.0f32;
    let mut max_relative_error = 0.0f32;
    let mut non_finite_values = 0usize;

    for (reference, candidate) in reference.iter().zip(candidate) {
        let reference = reference.to_f32();
        let candidate = candidate.to_f32();
        if !reference.is_finite() || !candidate.is_finite() {
            non_finite_values = non_finite_values.saturating_add(1);
            continue;
        }
        let error = candidate - reference;
        let abs_error = error.abs();
        max_abs_error = max_abs_error.max(abs_error);
        max_relative_error = max_relative_error.max(abs_error / reference.abs().max(1.0e-6));
        let reference = f64::from(reference);
        let candidate = f64::from(candidate);
        let error = f64::from(error);
        squared_error += error * error;
        squared_reference += reference * reference;
        squared_candidate += candidate * candidate;
        dot += reference * candidate;
    }

    let values = reference.len();
    let rms_error = (squared_error / values as f64).sqrt();
    let nrmse = if squared_reference > 0.0 {
        (squared_error / squared_reference).sqrt()
    } else if squared_error == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    let cosine_denominator = (squared_reference * squared_candidate).sqrt();
    let cosine = if cosine_denominator > 0.0 {
        dot / cosine_denominator
    } else if squared_reference == squared_candidate {
        1.0
    } else {
        0.0
    };
    let output_rms_ratio = if squared_reference > 0.0 {
        (squared_candidate / squared_reference).sqrt()
    } else {
        1.0
    };

    Ok(GemmErrorMetrics {
        values,
        nrmse,
        cosine,
        max_abs_error,
        max_relative_error,
        rms_error,
        output_rms_ratio,
        non_finite_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_has_identity_metrics() -> Result<()> {
        let values = [1.0, -2.0, 3.0].map(bf16::from_f32);
        let metrics = gemm_error_metrics(&values, &values)?;
        assert_eq!(metrics.nrmse, 0.0);
        assert!((metrics.cosine - 1.0).abs() < 1.0e-12);
        assert_eq!(metrics.output_rms_ratio, 1.0);
        Ok(())
    }
}
