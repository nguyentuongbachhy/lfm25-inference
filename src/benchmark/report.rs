use serde::Serialize;

#[derive(Debug, Clone, Copy)]
pub struct RequestObservation {
    pub ttft_ms: f64,
    pub tpot_ms: f64,
    pub queue_delay_ms: f64,
    pub decode_gpu_ms: f64,
    pub decode_submit_cpu_ms: f64,
    pub prompt_tokens: usize,
    pub output_tokens: usize,
    pub accepted: bool,
}

#[derive(Debug, Serialize)]
pub struct ServingSummary {
    pub offered_requests: usize,
    pub accepted_requests: usize,
    pub slo_requests: usize,
    pub ttft_p50_ms: Option<f64>,
    pub ttft_p95_ms: Option<f64>,
    pub ttft_p99_ms: Option<f64>,
    pub tpot_p50_ms: Option<f64>,
    pub tpot_p95_ms: Option<f64>,
    pub tpot_p99_ms: Option<f64>,
    pub queue_delay_p50_ms: Option<f64>,
    pub queue_delay_p95_ms: Option<f64>,
    pub decode_gpu_ms_per_token_mean: Option<f64>,
    pub decode_gpu_ms_per_token_p50: Option<f64>,
    pub decode_gpu_ms_per_token_p95: Option<f64>,
    pub decode_submit_cpu_ms_per_token_mean: Option<f64>,
    pub decode_submit_cpu_ms_per_token_p50: Option<f64>,
    pub decode_submit_cpu_ms_per_token_p95: Option<f64>,
    pub wall_seconds: f64,
    pub accepted_requests_per_second: f64,
    pub prompt_tokens_per_second: f64,
    pub output_tokens_per_second: f64,
    pub output_tokens: usize,
    pub goodput_tokens: usize,
    pub goodput_tokens_per_second: f64,
}

impl ServingSummary {
    pub fn from_observations(
        observations: &[RequestObservation],
        ttft_slo_ms: f64,
        tpot_slo_ms: f64,
        wall_seconds: f64,
    ) -> Self {
        let wall_seconds = wall_seconds.max(f64::MIN_POSITIVE);
        let mut ttft: Vec<f64> = observations
            .iter()
            .filter(|item| item.accepted)
            .map(|item| item.ttft_ms)
            .collect();
        let mut tpot: Vec<f64> = observations
            .iter()
            .filter(|item| item.accepted)
            .map(|item| item.tpot_ms)
            .collect();
        let mut queue_delay: Vec<f64> = observations
            .iter()
            .filter(|item| item.accepted)
            .map(|item| item.queue_delay_ms)
            .collect();
        let mut decode_gpu_ms_per_token = Vec::new();
        let mut decode_submit_cpu_ms_per_token = Vec::new();
        for item in observations.iter().filter(|item| item.accepted) {
            let decode_tokens = item.output_tokens.saturating_sub(1);
            if decode_tokens == 0 {
                continue;
            }
            let decode_tokens = decode_tokens as f64;
            let gpu = item.decode_gpu_ms / decode_tokens;
            let submit = item.decode_submit_cpu_ms / decode_tokens;
            if gpu.is_finite() {
                decode_gpu_ms_per_token.push(gpu);
            }
            if submit.is_finite() {
                decode_submit_cpu_ms_per_token.push(submit);
            }
        }
        ttft.sort_unstable_by(f64::total_cmp);
        tpot.sort_unstable_by(f64::total_cmp);
        queue_delay.sort_unstable_by(f64::total_cmp);
        decode_gpu_ms_per_token.sort_unstable_by(f64::total_cmp);
        decode_submit_cpu_ms_per_token.sort_unstable_by(f64::total_cmp);
        let accepted_requests = ttft.len();
        let slo_requests = observations
            .iter()
            .filter(|item| {
                item.accepted && item.ttft_ms < ttft_slo_ms && item.tpot_ms < tpot_slo_ms
            })
            .count();
        let output_tokens = observations
            .iter()
            .filter(|item| item.accepted)
            .map(|item| item.output_tokens)
            .sum();
        let prompt_tokens = observations
            .iter()
            .filter(|item| item.accepted)
            .map(|item| item.prompt_tokens)
            .sum::<usize>();
        let goodput_tokens = observations
            .iter()
            .filter(|item| {
                item.accepted && item.ttft_ms < ttft_slo_ms && item.tpot_ms < tpot_slo_ms
            })
            .map(|item| item.output_tokens)
            .sum();
        Self {
            offered_requests: observations.len(),
            accepted_requests,
            slo_requests,
            ttft_p50_ms: percentile(&ttft, 0.50),
            ttft_p95_ms: percentile(&ttft, 0.95),
            ttft_p99_ms: percentile(&ttft, 0.99),
            tpot_p50_ms: percentile(&tpot, 0.50),
            tpot_p95_ms: percentile(&tpot, 0.95),
            tpot_p99_ms: percentile(&tpot, 0.99),
            queue_delay_p50_ms: percentile(&queue_delay, 0.50),
            queue_delay_p95_ms: percentile(&queue_delay, 0.95),
            decode_gpu_ms_per_token_mean: mean(&decode_gpu_ms_per_token),
            decode_gpu_ms_per_token_p50: percentile(&decode_gpu_ms_per_token, 0.50),
            decode_gpu_ms_per_token_p95: percentile(&decode_gpu_ms_per_token, 0.95),
            decode_submit_cpu_ms_per_token_mean: mean(&decode_submit_cpu_ms_per_token),
            decode_submit_cpu_ms_per_token_p50: percentile(
                &decode_submit_cpu_ms_per_token,
                0.50,
            ),
            decode_submit_cpu_ms_per_token_p95: percentile(
                &decode_submit_cpu_ms_per_token,
                0.95,
            ),
            wall_seconds,
            accepted_requests_per_second: accepted_requests as f64 / wall_seconds,
            prompt_tokens_per_second: prompt_tokens as f64 / wall_seconds,
            output_tokens_per_second: output_tokens as f64 / wall_seconds,
            output_tokens,
            goodput_tokens,
            goodput_tokens_per_second: goodput_tokens as f64 / wall_seconds,
        }
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn percentile(sorted: &[f64], quantile: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let index = ((sorted.len() - 1) as f64 * quantile).ceil() as usize;
    sorted.get(index).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goodput_excludes_slo_violations_and_rejections() {
        let result = ServingSummary::from_observations(
            &[
                RequestObservation {
                    ttft_ms: 10.0,
                    tpot_ms: 8.0,
                    queue_delay_ms: 1.0,
                    decode_gpu_ms: 90.0,
                    decode_submit_cpu_ms: 9.0,
                    prompt_tokens: 4,
                    output_tokens: 10,
                    accepted: true,
                },
                RequestObservation {
                    ttft_ms: 500.0,
                    tpot_ms: 8.0,
                    queue_delay_ms: 2.0,
                    decode_gpu_ms: 180.0,
                    decode_submit_cpu_ms: 18.0,
                    prompt_tokens: 4,
                    output_tokens: 10,
                    accepted: true,
                },
                RequestObservation {
                    ttft_ms: 0.0,
                    tpot_ms: 0.0,
                    queue_delay_ms: 0.0,
                    decode_gpu_ms: 0.0,
                    decode_submit_cpu_ms: 0.0,
                    prompt_tokens: 0,
                    output_tokens: 0,
                    accepted: false,
                },
            ],
            400.0,
            50.0,
            1.0,
        );
        assert_eq!(result.goodput_tokens, 10);
        assert_eq!(result.slo_requests, 1);
        assert_eq!(result.decode_gpu_ms_per_token_mean, Some(15.0));
        assert_eq!(result.decode_submit_cpu_ms_per_token_mean, Some(1.5));
    }
}
