use serde::{Deserialize, Serialize};

use crate::tensor::Tensor;

use super::fp8_analysis::{Fp8GemmErrorReport, LocalPrecisionRecommendation};

const E4M3_MAX_FINITE: f32 = 448.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScaleStrategy {
    Amax,
    P99_99,
    P99_9,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ScalarScale {
    pub clip_amax: f32,
    pub quantize_multiplier: f32,
    pub dequantize_multiplier: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TensorwideE4m3ScaleCandidates {
    pub max: ScalarScale,
    pub p99: ScalarScale,
    pub p99_9: ScalarScale,
    pub p99_99: ScalarScale,
}

impl TensorwideE4m3ScaleCandidates {
    pub fn get(self, strategy: ScaleStrategy) -> ScalarScale {
        match strategy {
            ScaleStrategy::Amax => self.max,
            ScaleStrategy::P99_99 => self.p99_99,
            ScaleStrategy::P99_9 => self.p99_9,
        }
    }
}

pub(crate) struct Fp8LinearWeight {
    pub data: Tensor<u8>,
    pub activation_scale: ScalarScale,
    pub weight_scale: ScalarScale,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fp8SitePolicy {
    pub site: String,
    pub enabled: bool,
    pub activation_strategy: ScaleStrategy,
    pub weight_strategy: ScaleStrategy,
    pub activation_scale: ScalarScale,
    pub weight_scale: ScalarScale,
    pub local_nrmse: f64,
    pub local_cosine: f64,
    pub expected_decode_saving_us: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fp8PrecisionPolicy {
    pub schema_version: u32,
    pub name: String,
    pub source: String,
    pub decode_only: bool,
    pub sites: Vec<Fp8SitePolicy>,
}

impl Fp8PrecisionPolicy {
    pub(crate) fn from_local_errors(report: &Fp8GemmErrorReport, name: impl Into<String>) -> Self {
        let sites = report
            .sites
            .iter()
            .filter_map(|site| {
                let selected = site.selected_trial?;
                let trial = &site.trials[selected];
                Some(Fp8SitePolicy {
                    site: site.site.clone(),
                    enabled: site.recommendation == LocalPrecisionRecommendation::Fp8Tensorwide,
                    activation_strategy: trial.activation_strategy,
                    weight_strategy: trial.weight_strategy,
                    activation_scale: trial.activation_scale,
                    weight_scale: trial.weight_scale,
                    local_nrmse: trial.metrics.nrmse,
                    local_cosine: trial.metrics.cosine,
                    expected_decode_saving_us: expected_decode_saving_us(&site.site),
                })
            })
            .collect();
        Self {
            schema_version: 1,
            name: name.into(),
            source: "real_checkpoint_teacher_forced_gemm_error".to_string(),
            decode_only: true,
            sites,
        }
    }

    pub(crate) fn retain_class(&mut self, class: PrecisionClass) {
        for site in &mut self.sites {
            site.enabled &= class.includes(&site.site);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PrecisionClass {
    Mlp,
    MlpLmHead,
    MlpLmHeadConv,
    All,
}

impl PrecisionClass {
    fn includes(self, site: &str) -> bool {
        let mlp = site.contains(".mlp.");
        let lm_head = site == "lm_head";
        let conv = site.contains(".conv.");
        match self {
            Self::Mlp => mlp,
            Self::MlpLmHead => mlp || lm_head,
            Self::MlpLmHeadConv => mlp || lm_head || conv,
            Self::All => true,
        }
    }
}

fn expected_decode_saving_us(site: &str) -> f64 {
    if site.ends_with("mlp.gate_up") {
        (3415.0 - 3415.0 / 1.863) / 16.0
    } else if site.ends_with("mlp.down") {
        (1801.0 - 1801.0 / 1.546) / 16.0
    } else if site == "lm_head" {
        1007.0 - 1007.0 / 2.41
    } else if site.contains(".conv.") {
        (1018.0 + 442.0) * 0.20 / 20.0
    } else if site.contains(".attention.") {
        (396.0 + 195.0) * 0.18 / 24.0
    } else {
        0.0
    }
}

pub(crate) fn tensorwide_e4m3_scale_candidates(
    amax: f32,
    p99: f32,
    p99_9: f32,
    p99_99: f32,
) -> TensorwideE4m3ScaleCandidates {
    TensorwideE4m3ScaleCandidates {
        max: scalar_scale(amax),
        p99: scalar_scale(p99),
        p99_9: scalar_scale(p99_9),
        p99_99: scalar_scale(p99_99),
    }
}

fn scalar_scale(clip_amax: f32) -> ScalarScale {
    if !clip_amax.is_finite() || clip_amax <= 0.0 {
        return ScalarScale {
            clip_amax: 0.0,
            quantize_multiplier: 1.0,
            dequantize_multiplier: 1.0,
        };
    }

    ScalarScale {
        clip_amax,
        quantize_multiplier: E4M3_MAX_FINITE / clip_amax,
        dequantize_multiplier: clip_amax / E4M3_MAX_FINITE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_scale_round_trips() {
        let candidates = tensorwide_e4m3_scale_candidates(2.0, 1.0, 1.5, 1.75);
        assert!((candidates.max.quantize_multiplier - 224.0).abs() < f32::EPSILON);
        assert!(
            (candidates.max.quantize_multiplier * candidates.max.dequantize_multiplier - 1.0).abs()
                < 1e-6
        );
    }

    #[test]
    fn zero_scale_is_identity() {
        let candidates = tensorwide_e4m3_scale_candidates(0.0, 0.0, 0.0, 0.0);
        assert_eq!(candidates.max.quantize_multiplier, 1.0);
        assert_eq!(candidates.max.dequantize_multiplier, 1.0);
    }
}
