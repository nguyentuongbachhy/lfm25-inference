use std::{fs, path::Path};

use anyhow::{Context as _, Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CostPoint {
    pub batch: usize,
    pub tokens: usize,
    pub context: usize,
    pub milliseconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostCurve {
    pub points: Vec<CostPoint>,
}

impl CostCurve {
    pub fn new(points: Vec<CostPoint>) -> Result<Self> {
        ensure!(!points.is_empty(), "cost curve requires measured points");
        ensure!(
            points.iter().all(|point| point.batch > 0
                && point.tokens > 0
                && point.context > 0
                && point.milliseconds.is_finite()
                && point.milliseconds >= 0.0),
            "cost curve contains invalid point"
        );
        Ok(Self { points })
    }

    pub fn predict(&self, batch: usize, tokens: usize, context: usize) -> f64 {
        let batch = batch.max(1);
        let tokens = tokens.max(1);
        let context = context.max(1);
        if let Some(point) = self.points.iter().find(|point| {
            point.batch == batch && point.tokens == tokens && point.context == context
        }) {
            return point.milliseconds;
        }
        let mut dominating = None;
        let mut best_distance = f64::INFINITY;
        for point in &self.points {
            if point.batch < batch || point.tokens < tokens || point.context < context {
                continue;
            }
            let distance = (point.batch as f64 / batch as f64).ln()
                + (point.tokens as f64 / tokens as f64).ln()
                + (point.context as f64 / context as f64).ln();
            if distance < best_distance
                || (distance == best_distance
                    && dominating
                        .is_none_or(|best: CostPoint| point.milliseconds > best.milliseconds))
            {
                best_distance = distance;
                dominating = Some(*point);
            }
        }
        if let Some(point) = dominating {
            return point.milliseconds;
        }

        // Outside the measured envelope, extrapolate conservatively from the
        // nearest point. Multiplying each dimension's expansion ratio avoids
        // inventing target-specific scaling exponents in the scheduler.
        let mut nearest = self.points[0];
        best_distance = f64::INFINITY;
        for point in &self.points {
            let distance = (batch as f64 / point.batch as f64).ln().abs()
                + (tokens as f64 / point.tokens as f64).ln().abs()
                + (context as f64 / point.context as f64).ln().abs();
            if distance < best_distance {
                best_distance = distance;
                nearest = *point;
            }
        }
        let batch_scale = (batch as f64 / nearest.batch as f64).max(1.0);
        let token_scale = (tokens as f64 / nearest.tokens as f64).max(1.0);
        let context_scale = (context as f64 / nearest.context as f64).max(1.0);
        nearest.milliseconds * batch_scale * token_scale * context_scale
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareCostModel {
    pub schema_version: u32,
    pub gpu_name: String,
    pub page_size: usize,
    pub decode_bf16: CostCurve,
    pub decode_fp8: CostCurve,
    pub prefill_bf16: CostCurve,
    pub interactive_prompt_limit: usize,
    pub ttft_slo_ms: f64,
    pub tpot_slo_ms: f64,
}

impl HardwareCostModel {
    pub fn from_path(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read hardware profile {}", path.display()))?;
        let profile: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse hardware profile {}", path.display()))?;
        ensure!(
            profile.schema_version == 1,
            "unsupported hardware profile schema {}",
            profile.schema_version
        );
        ensure!(
            profile.ttft_slo_ms.is_finite()
                && profile.ttft_slo_ms > 0.0
                && profile.tpot_slo_ms.is_finite()
                && profile.tpot_slo_ms > 0.0,
            "hardware profile has invalid SLO values"
        );
        ensure!(
            matches!(profile.page_size, 16 | 32),
            "hardware profile has invalid page size {}",
            profile.page_size
        );
        Ok(profile)
    }

    pub fn predict_decode_ms(&self, batch: usize, maximum_context: usize, fp8: bool) -> f64 {
        let curve = if fp8 {
            &self.decode_fp8
        } else {
            &self.decode_bf16
        };
        curve.predict(batch, 1, maximum_context)
    }

    pub fn predict_prefill_ms(&self, tokens: usize) -> f64 {
        self.prefill_bf16.predict(1, tokens, tokens)
    }

    pub fn largest_prefill_chunk(&self, remaining: usize, budget_ms: f64) -> usize {
        if remaining == 0 || budget_ms <= 0.0 {
            return 0;
        }
        for candidate in [1024usize, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1] {
            if candidate > remaining {
                continue;
            }
            if self.predict_prefill_ms(candidate) <= budget_ms {
                return candidate;
            }
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measured_points_are_exact_and_unmeasured_interior_is_conservative() -> Result<()> {
        let curve = CostCurve::new(vec![
            CostPoint {
                batch: 1,
                tokens: 1,
                context: 128,
                milliseconds: 7.0,
            },
            CostPoint {
                batch: 8,
                tokens: 1,
                context: 128,
                milliseconds: 8.0,
            },
        ])?;
        assert_eq!(curve.predict(1, 1, 128), 7.0);
        assert_eq!(curve.predict(4, 1, 128), 8.0);
        assert!(curve.predict(16, 1, 128) >= 16.0);
        Ok(())
    }
}
