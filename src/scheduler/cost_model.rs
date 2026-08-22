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

    /// Decode profiles are measured on a dense batch/context grid. Interpolate
    /// inside that measured grid instead of jumping to the next dominating
    /// power-of-two point. This keeps exact measurements exact while avoiding
    /// artificial B32->B33 and context 2048->2049 latency cliffs in admission
    /// and scheduling. Outside the measured envelope we still extrapolate
    /// conservatively from the nearest measured boundary.
    pub fn predict_decode_interpolated(&self, batch: usize, context: usize) -> f64 {
        let batch = batch.max(1);
        let context = context.max(1);

        if let Some(point) = self
            .points
            .iter()
            .find(|point| point.tokens == 1 && point.batch == batch && point.context == context)
        {
            return point.milliseconds;
        }

        let mut lower_context = None;
        let mut upper_context = None;
        for point in &self.points {
            if point.tokens != 1 {
                continue;
            }
            if point.context <= context && lower_context.is_none_or(|value| point.context > value) {
                lower_context = Some(point.context);
            }
            if point.context >= context && upper_context.is_none_or(|value| point.context < value) {
                upper_context = Some(point.context);
            }
        }

        match (lower_context, upper_context) {
            (Some(lower), Some(upper)) if lower == upper => self
                .interpolate_decode_batch(batch, lower)
                .unwrap_or_else(|| self.predict(batch, 1, context)),
            (Some(lower), Some(upper)) => {
                let lower_ms = self
                    .interpolate_decode_batch(batch, lower)
                    .unwrap_or_else(|| self.predict(batch, 1, lower));
                let upper_ms = self
                    .interpolate_decode_batch(batch, upper)
                    .unwrap_or_else(|| self.predict(batch, 1, upper));
                let span = upper - lower;
                let offset = context - lower;
                let weight = offset as f64 / span as f64;
                lower_ms + (upper_ms - lower_ms) * weight
            }
            (Some(lower), None) => {
                let lower_ms = self
                    .interpolate_decode_batch(batch, lower)
                    .unwrap_or_else(|| self.predict(batch, 1, lower));
                lower_ms * (context as f64 / lower as f64).max(1.0)
            }
            (None, Some(upper)) => self
                .interpolate_decode_batch(batch, upper)
                .unwrap_or_else(|| self.predict(batch, 1, upper)),
            (None, None) => self.predict(batch, 1, context),
        }
    }

    /// Prefill profiles are single-segment token curves. Dense profiles are
    /// interpolated between neighboring measured token counts so scheduler
    /// admission does not jump from (for example) the 128-token cost straight
    /// to the 512-token cost. Exact measurements remain exact. Outside the
    /// measured envelope, fall back to the conservative generic predictor.
    pub fn predict_prefill_interpolated(&self, tokens: usize) -> f64 {
        if tokens == 0 {
            return 0.0;
        }

        if let Some(point) = self.points.iter().find(|point| {
            point.batch == 1 && point.tokens == tokens && point.context == tokens
        }) {
            return point.milliseconds;
        }

        let mut lower = None;
        let mut upper = None;
        for point in &self.points {
            if point.batch != 1 || point.tokens != point.context {
                continue;
            }
            if point.tokens < tokens
                && lower.is_none_or(|best: CostPoint| point.tokens > best.tokens)
            {
                lower = Some(*point);
            }
            if point.tokens > tokens
                && upper.is_none_or(|best: CostPoint| point.tokens < best.tokens)
            {
                upper = Some(*point);
            }
        }

        match (lower, upper) {
            (Some(lower), Some(upper)) => {
                let span = upper.tokens - lower.tokens;
                let offset = tokens - lower.tokens;
                let weight = offset as f64 / span as f64;
                lower.milliseconds + (upper.milliseconds - lower.milliseconds) * weight
            }
            _ => self.predict(1, tokens, tokens),
        }
    }

    fn interpolate_decode_batch(&self, batch: usize, context: usize) -> Option<f64> {
        let mut lower = None;
        let mut upper = None;
        for point in &self.points {
            if point.tokens != 1 || point.context != context {
                continue;
            }
            if point.batch == batch {
                return Some(point.milliseconds);
            }
            if point.batch < batch && lower.is_none_or(|best: CostPoint| point.batch > best.batch) {
                lower = Some(*point);
            }
            if point.batch > batch && upper.is_none_or(|best: CostPoint| point.batch < best.batch) {
                upper = Some(*point);
            }
        }

        match (lower, upper) {
            (Some(lower), Some(upper)) => {
                let span = upper.batch - lower.batch;
                let offset = batch - lower.batch;
                let weight = offset as f64 / span as f64;
                Some(lower.milliseconds + (upper.milliseconds - lower.milliseconds) * weight)
            }
            (Some(lower), None) => {
                Some(lower.milliseconds * (batch as f64 / lower.batch as f64).max(1.0))
            }
            (None, Some(upper)) => Some(upper.milliseconds),
            (None, None) => None,
        }
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
        curve.predict_decode_interpolated(batch, maximum_context)
    }

    pub fn predict_prefill_ms(&self, tokens: usize) -> f64 {
        self.prefill_bf16.predict_prefill_interpolated(tokens)
    }

    /// Find the largest page-aligned aggregate prefill token count that fits a
    /// latency budget. The search is deliberately tiny (<= 32 candidates for
    /// PS16 with the current 512-token ceiling), so a measured/interpolated LUT
    /// remains simpler and more deterministic than fitting an analytic model.
    pub fn max_prefill_tokens_within_budget(
        &self,
        budget_ms: f64,
        hard_limit: usize,
    ) -> usize {
        if !budget_ms.is_finite() || budget_ms <= 0.0 || hard_limit == 0 {
            return 0;
        }
        let quantum = self.page_size.max(1);
        let mut best = 0usize;
        let mut candidate = quantum;
        while candidate <= hard_limit {
            if self.predict_prefill_ms(candidate) <= budget_ms {
                best = candidate;
            }
            let Some(next) = candidate.checked_add(quantum) else {
                break;
            };
            candidate = next;
        }
        best
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

    #[test]
    fn decode_interpolation_preserves_measurements_and_removes_batch_cliffs() -> Result<()> {
        let curve = CostCurve::new(vec![
            CostPoint {
                batch: 16,
                tokens: 1,
                context: 2048,
                milliseconds: 11.0,
            },
            CostPoint {
                batch: 32,
                tokens: 1,
                context: 2048,
                milliseconds: 16.0,
            },
            CostPoint {
                batch: 64,
                tokens: 1,
                context: 2048,
                milliseconds: 24.0,
            },
            CostPoint {
                batch: 16,
                tokens: 1,
                context: 8192,
                milliseconds: 22.0,
            },
        ])?;
        assert_eq!(curve.predict_decode_interpolated(32, 2048), 16.0);
        let batch_33 = curve.predict_decode_interpolated(33, 2048);
        assert!(batch_33 > 16.0 && batch_33 < 24.0);
        let context_2049 = curve.predict_decode_interpolated(16, 2049);
        assert!(context_2049 > 11.0 && context_2049 < 22.0);
        Ok(())
    }

    #[test]
    fn decode_interpolation_extrapolates_conservatively_outside_grid() -> Result<()> {
        let curve = CostCurve::new(vec![CostPoint {
            batch: 16,
            tokens: 1,
            context: 2048,
            milliseconds: 12.0,
        }])?;
        assert!(curve.predict_decode_interpolated(32, 2048) >= 24.0);
        assert!(curve.predict_decode_interpolated(16, 4096) >= 24.0);
        Ok(())
    }

    #[test]
    fn prefill_interpolation_preserves_dense_measurements_and_removes_token_cliffs() -> Result<()> {
        let curve = CostCurve::new(vec![
            CostPoint {
                batch: 1,
                tokens: 128,
                context: 128,
                milliseconds: 14.0,
            },
            CostPoint {
                batch: 1,
                tokens: 256,
                context: 256,
                milliseconds: 24.0,
            },
            CostPoint {
                batch: 1,
                tokens: 512,
                context: 512,
                milliseconds: 47.0,
            },
        ])?;
        assert_eq!(curve.predict_prefill_interpolated(128), 14.0);
        assert_eq!(curve.predict_prefill_interpolated(256), 24.0);
        let token_129 = curve.predict_prefill_interpolated(129);
        assert!(token_129 > 14.0 && token_129 < 24.0);
        let token_384 = curve.predict_prefill_interpolated(384);
        assert!(token_384 > 24.0 && token_384 < 47.0);
        Ok(())
    }

    #[test]
    fn prefill_budget_solver_returns_largest_page_aligned_safe_chunk() -> Result<()> {
        let model = HardwareCostModel {
            schema_version: 1,
            gpu_name: "test".into(),
            page_size: 16,
            decode_bf16: CostCurve::new(vec![CostPoint {
                batch: 1,
                tokens: 1,
                context: 128,
                milliseconds: 6.0,
            }])?,
            decode_fp8: CostCurve::new(vec![CostPoint {
                batch: 1,
                tokens: 1,
                context: 128,
                milliseconds: 5.0,
            }])?,
            prefill_bf16: CostCurve::new(vec![
                CostPoint {
                    batch: 1,
                    tokens: 128,
                    context: 128,
                    milliseconds: 14.0,
                },
                CostPoint {
                    batch: 1,
                    tokens: 256,
                    context: 256,
                    milliseconds: 24.0,
                },
                CostPoint {
                    batch: 1,
                    tokens: 384,
                    context: 384,
                    milliseconds: 34.0,
                },
                CostPoint {
                    batch: 1,
                    tokens: 512,
                    context: 512,
                    milliseconds: 47.0,
                },
            ])?,
            interactive_prompt_limit: 2048,
            ttft_slo_ms: 400.0,
            tpot_slo_ms: 50.0,
        };
        assert_eq!(model.max_prefill_tokens_within_budget(24.0, 512), 256);
        assert_eq!(model.max_prefill_tokens_within_budget(33.9, 512), 368);
        assert_eq!(model.max_prefill_tokens_within_budget(5.0, 512), 0);
        Ok(())
    }
}