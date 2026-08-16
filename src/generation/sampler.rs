use std::collections::HashSet;

use anyhow::{Result, ensure};
use half::bf16;

use crate::{cuda::CudaRuntime, ops, tensor::Tensor};

#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 50,
            repetition_penalty: 1.0,
            seed: 0x4c_46_4d_32,
        }
    }
}

impl SamplingConfig {
    pub fn validate(self) -> Result<Self> {
        ensure!(
            self.temperature.is_finite() && self.temperature >= 0.0,
            "temperature must be finite and non-negative"
        );
        ensure!(self.top_k > 0, "top_k must be positive");
        ensure!(
            self.repetition_penalty.is_finite() && self.repetition_penalty >= 1.0,
            "repetition_penalty must be finite and >= 1"
        );
        Ok(self)
    }
}

pub struct Sampler {
    config: SamplingConfig,
    random_state: u64,
}

impl Sampler {
    pub fn new(config: SamplingConfig) -> Result<Self> {
        let config = config.validate()?;
        let random_state = if config.seed == 0 {
            0x9e3779b97f4a7c15
        } else {
            config.seed
        };
        Ok(Self {
            config,
            random_state,
        })
    }

    pub fn sample(
        &mut self,
        runtime: &CudaRuntime,
        logits: &Tensor<bf16>,
        history: &[u32],
    ) -> Result<u32> {
        ensure!(
            logits.rank() == 2 && logits.dims()[0] == 1,
            "sampler expects logits [1,vocab], got {:?}",
            logits.dims()
        );

        if self.config.temperature == 0.0 && self.config.repetition_penalty == 1.0 {
            let token = ops::argmax_bf16(runtime, logits)?;
            return Ok(runtime.download(&token)?[0]);
        }

        let host = runtime.download(logits)?;
        let repeated: HashSet<u32> = history.iter().copied().collect();
        let mut candidates: Vec<(u32, f32)> = host
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let mut score = value.to_f32();
                if repeated.contains(&(index as u32)) {
                    score = if score >= 0.0 {
                        score / self.config.repetition_penalty
                    } else {
                        score * self.config.repetition_penalty
                    };
                }
                (index as u32, score)
            })
            .collect();

        if self.config.temperature == 0.0 {
            return candidates
                .into_iter()
                .max_by(|left, right| left.1.total_cmp(&right.1))
                .map(|candidate| candidate.0)
                .ok_or_else(|| anyhow::anyhow!("sampler received empty logits"));
        }

        let keep = self.config.top_k.min(candidates.len());
        if keep < candidates.len() {
            candidates.select_nth_unstable_by(keep - 1, |left, right| right.1.total_cmp(&left.1));
            candidates.truncate(keep);
        }

        let maximum = candidates
            .iter()
            .map(|candidate| candidate.1)
            .max_by(f32::total_cmp)
            .ok_or_else(|| anyhow::anyhow!("sampler received empty logits"))?;
        let mut total = 0.0f64;
        let probabilities: Vec<f64> = candidates
            .iter()
            .map(|candidate| {
                let probability =
                    f64::from(((candidate.1 - maximum) / self.config.temperature).exp());
                total += probability;
                probability
            })
            .collect();
        ensure!(
            total.is_finite() && total > 0.0,
            "sampling probabilities are invalid"
        );

        let mut threshold = self.next_unit_f64() * total;
        for (candidate, probability) in candidates.iter().zip(probabilities) {
            if threshold <= probability {
                return Ok(candidate.0);
            }
            threshold -= probability;
        }
        Ok(candidates[candidates.len() - 1].0)
    }

    fn next_unit_f64(&mut self) -> f64 {
        let mut value = self.random_state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.random_state = value;
        (value >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_config_rejects_invalid_values() {
        assert!(
            SamplingConfig {
                temperature: -1.0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            SamplingConfig {
                top_k: 0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            SamplingConfig {
                repetition_penalty: 0.9,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn xorshift_is_deterministic_for_a_seed() -> Result<()> {
        let config = SamplingConfig {
            seed: 42,
            ..Default::default()
        };
        let mut left = Sampler::new(config)?;
        let mut right = Sampler::new(config)?;
        for _ in 0..16 {
            assert_eq!(left.next_unit_f64(), right.next_unit_f64());
        }
        Ok(())
    }
}
