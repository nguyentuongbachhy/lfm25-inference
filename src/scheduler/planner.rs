use anyhow::{Result, ensure};

use super::{HardwareCostModel, RequestPhase, RequestSlotId, RequestSlots};

const DECODE_TAIL_CONTEXT_THRESHOLD: usize = 1536;
const DECODE_TAIL_QUEUE_THRESHOLD: usize = 24;
const DEADLINE_SAFETY_MARGIN_US: u64 = 2_000;

#[derive(Debug, Clone, Copy)]
pub struct SchedulerConfig {
    pub step_budget_ms: f64,
    pub ttft_slo_us: u64,
    pub tpot_slo_us: u64,
    pub fp8_decode: bool,
    /// Hard ceiling only. The scheduler chooses a smaller page-aligned amount
    /// from the measured latency budget on every step.
    pub maximum_prefill_tokens: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            step_budget_ms: 40.0,
            ttft_slo_us: 400_000,
            tpot_slo_us: 50_000,
            fp8_decode: true,
            maximum_prefill_tokens: 512,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduledWork {
    Decode { slot: RequestSlotId },
    Prefill { slot: RequestSlotId, tokens: usize },
}

#[derive(Debug)]
pub struct BatchPlan {
    work: Vec<ScheduledWork>,
    pub predicted_ms: f64,
    pub decode_count: usize,
    pub prefill_tokens: usize,
    /// Budget after applying the configured cap and the nearest decode token
    /// deadline. This is intentionally surfaced for scheduler telemetry.
    pub effective_budget_ms: f64,
    pub earliest_decode_slack_ms: Option<f64>,
    pub deadline_limited: bool,
    /// Number of decode requests tied for the minimum deadline slack. Tracking
    /// this lets load benchmarks distinguish broad SLO pressure from a single
    /// outlier request throttling an otherwise healthy batch.
    pub deadline_limiter_count: usize,
}

impl BatchPlan {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            work: Vec::with_capacity(capacity),
            predicted_ms: 0.0,
            decode_count: 0,
            prefill_tokens: 0,
            effective_budget_ms: 0.0,
            earliest_decode_slack_ms: None,
            deadline_limited: false,
            deadline_limiter_count: 0,
        }
    }

    pub fn work(&self) -> &[ScheduledWork] {
        &self.work
    }

    pub fn deadline_limited_by_single_request(&self) -> bool {
        self.deadline_limited && self.deadline_limiter_count == 1
    }
}

pub struct Scheduler {
    config: SchedulerConfig,
    cost: HardwareCostModel,
    plan: BatchPlan,
    prefill_order: Vec<(u64, RequestSlotId)>,
}

impl Scheduler {
    pub fn new(capacity: usize, config: SchedulerConfig, cost: HardwareCostModel) -> Result<Self> {
        ensure!(capacity > 0, "scheduler capacity must be positive");
        ensure!(
            config.step_budget_ms.is_finite() && config.step_budget_ms > 0.0,
            "scheduler step budget must be positive"
        );
        Ok(Self {
            config,
            cost,
            plan: BatchPlan::with_capacity(capacity.saturating_add(1)),
            prefill_order: Vec::with_capacity(capacity),
        })
    }

    pub fn schedule(&mut self, requests: &RequestSlots, now_us: u64) -> &BatchPlan {
        self.plan.work.clear();
        self.plan.predicted_ms = 0.0;
        self.plan.decode_count = 0;
        self.plan.prefill_tokens = 0;
        self.plan.effective_budget_ms = self.config.step_budget_ms;
        self.plan.earliest_decode_slack_ms = None;
        self.plan.deadline_limited = false;
        self.plan.deadline_limiter_count = 0;
        self.prefill_order.clear();

        let mut maximum_context = 1usize;
        let mut earliest_decode_slack_us = u64::MAX;
        let mut deadline_limiter_count = 0usize;
        for (index, request) in requests.entries().iter().enumerate() {
            if request.phase == RequestPhase::Decoding {
                self.plan.work.push(ScheduledWork::Decode {
                    slot: RequestSlotId(index as u32),
                });
                maximum_context = maximum_context.max(request.tokens().len());
                let slack_us = request.next_token_deadline_us.saturating_sub(now_us);
                if slack_us < earliest_decode_slack_us {
                    earliest_decode_slack_us = slack_us;
                    deadline_limiter_count = 1;
                } else if slack_us == earliest_decode_slack_us {
                    deadline_limiter_count = deadline_limiter_count.saturating_add(1);
                }
            }
        }
        self.plan.work.sort_unstable_by_key(|work| match work {
            ScheduledWork::Decode { slot } => requests
                .get(*slot)
                .map(|request| request.next_token_deadline_us.saturating_sub(now_us))
                .unwrap_or(u64::MAX),
            ScheduledWork::Prefill { .. } => u64::MAX,
        });
        self.plan.decode_count = self.plan.work.len();

        if self.plan.decode_count > 0 {
            let earliest_slack_ms = earliest_decode_slack_us as f64 / 1000.0;
            let deadline_budget_ms = earliest_decode_slack_us
                .saturating_sub(DEADLINE_SAFETY_MARGIN_US) as f64
                / 1000.0;
            self.plan.earliest_decode_slack_ms = Some(earliest_slack_ms);
            self.plan.effective_budget_ms = self.config.step_budget_ms.min(deadline_budget_ms);
            self.plan.deadline_limited =
                self.plan.effective_budget_ms + f64::EPSILON < self.config.step_budget_ms;
            self.plan.deadline_limiter_count = deadline_limiter_count;
        }

        let decode_ms = if self.plan.decode_count == 0 {
            0.0
        } else {
            self.cost.predict_decode_ms(
                self.plan.decode_count,
                maximum_context,
                self.config.fp8_decode,
            )
        };
        self.plan.predicted_ms = decode_ms;

        let remaining_budget = (self.plan.effective_budget_ms - decode_ms).max(0.0);
        if remaining_budget <= 0.0 || self.config.maximum_prefill_tokens == 0 {
            return &self.plan;
        }

        for (index, request) in requests.entries().iter().enumerate() {
            if request.phase != RequestPhase::QueuedPrefill
                || request.prefilled >= request.prompt_len
            {
                continue;
            }
            // `prefilled` includes attached/refreshed radix-cache prefixes, so
            // all scheduling below naturally operates on effective remaining
            // prompt work rather than raw prompt length.
            let urgency = request
                .first_token_deadline_us
                .saturating_sub(now_us)
                .saturating_sub(now_us.saturating_sub(request.arrival_us) / 4);
            self.prefill_order
                .push((urgency, RequestSlotId(index as u32)));
        }
        self.prefill_order.sort_unstable_by_key(|entry| entry.0);

        let page_size = self.cost.page_size;
        let tail_pressure = self.prefill_order.len() >= DECODE_TAIL_QUEUE_THRESHOLD
            && self.prefill_order.iter().any(|&(_, slot)| {
                requests.get(slot).is_ok_and(|request| {
                    let remaining = request.prompt_len.saturating_sub(request.prefilled);
                    request.prefilled >= DECODE_TAIL_CONTEXT_THRESHOLD
                        && remaining > 0
                        && remaining <= page_size
                })
            });

        if tail_pressure {
            // Keep this step decode-only. Any regular (>1-token) prefill segment
            // would make the whole ragged attention batch fall back to the
            // generic prefill path, defeating the purpose of singleton tails.
            let mut singleton_count = 0usize;
            let mut singleton_context = maximum_context;
            for &(_, slot) in &self.prefill_order {
                let Ok(request) = requests.get(slot) else {
                    continue;
                };
                let remaining = request.prompt_len.saturating_sub(request.prefilled);
                if request.prefilled == 0 || remaining == 0 || remaining > page_size {
                    continue;
                }

                let projected_count = self
                    .plan
                    .decode_count
                    .saturating_add(singleton_count)
                    .saturating_add(1);
                let projected_context = singleton_context.max(request.prefilled.saturating_add(1));
                let projected_ms = self.cost.predict_decode_ms(
                    projected_count,
                    projected_context,
                    self.config.fp8_decode,
                );
                if projected_ms >= self.plan.effective_budget_ms
                    || projected_ms * 1000.0 >= self.config.tpot_slo_us as f64
                {
                    break;
                }

                self.plan
                    .work
                    .push(ScheduledWork::Prefill { slot, tokens: 1 });
                self.plan.prefill_tokens = self.plan.prefill_tokens.saturating_add(1);
                singleton_count = singleton_count.saturating_add(1);
                singleton_context = projected_context;
                self.plan.predicted_ms = projected_ms;
            }
            return &self.plan;
        }

        let safe_prefill_total = self.cost.max_prefill_tokens_within_budget(
            remaining_budget,
            self.config.maximum_prefill_tokens,
        );
        if safe_prefill_total == 0 {
            return &self.plan;
        }

        for &(_, slot) in &self.prefill_order {
            if self.plan.prefill_tokens >= safe_prefill_total {
                break;
            }
            let Ok(request) = requests.get(slot) else {
                continue;
            };
            let remaining = request.prompt_len.saturating_sub(request.prefilled);
            if remaining == 0 {
                continue;
            }

            let available = safe_prefill_total.saturating_sub(self.plan.prefill_tokens);
            let capped = remaining.min(available);
            let chunk = if capped == remaining {
                // A final tail may be shorter than one KV page.
                capped
            } else {
                capped - (capped % page_size)
            };
            if chunk == 0 {
                continue;
            }

            self.plan.work.push(ScheduledWork::Prefill {
                slot,
                tokens: chunk,
            });
            self.plan.prefill_tokens = self.plan.prefill_tokens.saturating_add(chunk);
        }

        if self.plan.prefill_tokens > 0 {
            self.plan.predicted_ms =
                decode_ms + self.cost.predict_prefill_ms(self.plan.prefill_tokens);
        }
        &self.plan
    }
}

#[cfg(test)]
mod tests {
    use crate::scheduler::{CostCurve, CostPoint, RequestInit};

    use super::*;

    fn cost_model() -> Result<HardwareCostModel> {
        let decode_points = vec![
            CostPoint {
                batch: 1,
                tokens: 1,
                context: 128,
                milliseconds: 6.0,
            },
            CostPoint {
                batch: 64,
                tokens: 1,
                context: 2048,
                milliseconds: 20.0,
            },
        ];
        Ok(HardwareCostModel {
            schema_version: 1,
            gpu_name: "test".into(),
            page_size: 16,
            decode_bf16: CostCurve::new(decode_points.clone())?,
            decode_fp8: CostCurve::new(decode_points)?,
            prefill_bf16: CostCurve::new(vec![
                CostPoint {
                    batch: 1,
                    tokens: 16,
                    context: 16,
                    milliseconds: 5.0,
                },
                CostPoint {
                    batch: 1,
                    tokens: 32,
                    context: 32,
                    milliseconds: 6.0,
                },
                CostPoint {
                    batch: 1,
                    tokens: 64,
                    context: 64,
                    milliseconds: 8.0,
                },
                CostPoint {
                    batch: 1,
                    tokens: 128,
                    context: 128,
                    milliseconds: 12.0,
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
                    milliseconds: 48.0,
                },
            ])?,
            interactive_prompt_limit: 2048,
            ttft_slo_ms: 400.0,
            tpot_slo_ms: 50.0,
        })
    }

    #[test]
    fn decode_is_scheduled_before_aged_prefill_without_reallocation() -> Result<()> {
        let mut slots = RequestSlots::new(3, 1024)?;
        let decode = slots.acquire().expect("decode slot");
        slots.get_mut(decode)?.initialize(RequestInit::new(
            1,
            &[1],
            512,
            0,
            400_000,
            50_000,
            32,
        ))?;
        slots.get_mut(decode)?.phase = RequestPhase::Decoding;
        let prefill = slots.acquire().expect("prefill slot");
        slots
            .get_mut(prefill)?
            .initialize(RequestInit::new(2, &[2; 256], 512, 0, 400_000, 50_000, 32))?;
        let mut scheduler = Scheduler::new(3, SchedulerConfig::default(), cost_model()?)?;
        let capacity = scheduler.plan.work.capacity();
        let prefill_capacity = scheduler.prefill_order.capacity();
        let plan = scheduler.schedule(&slots, 10_000);
        assert!(matches!(plan.work()[0], ScheduledWork::Decode { .. }));
        assert!(plan.prefill_tokens > 0);
        assert_eq!(scheduler.plan.work.capacity(), capacity);
        assert_eq!(scheduler.prefill_order.capacity(), prefill_capacity);
        Ok(())
    }

    #[test]
    fn packs_multiple_short_prefills_into_one_step() -> Result<()> {
        let mut slots = RequestSlots::new(4, 1024)?;
        for request_id in 1..=3u64 {
            let slot = slots.acquire().expect("prefill slot");
            slots.get_mut(slot)?.initialize(RequestInit::new(
                request_id, &[2; 32], 128, 0, 400_000, 50_000, 8,
            ))?;
        }
        let mut scheduler = Scheduler::new(4, SchedulerConfig::default(), cost_model()?)?;
        let plan = scheduler.schedule(&slots, 10_000);
        let prefills = plan
            .work()
            .iter()
            .filter(|work| matches!(work, ScheduledWork::Prefill { .. }))
            .count();
        assert_eq!(prefills, 3);
        assert_eq!(plan.prefill_tokens, 96);
        assert!(plan.predicted_ms <= SchedulerConfig::default().step_budget_ms);
        Ok(())
    }

    #[test]
    fn adaptive_prefill_uses_measured_budget_instead_of_static_candidate_list() -> Result<()> {
        let mut slots = RequestSlots::new(1, 1024)?;
        let slot = slots.acquire().expect("prefill slot");
        slots.get_mut(slot)?.initialize(RequestInit::new(
            1,
            &[2; 512],
            640,
            0,
            400_000,
            50_000,
            40,
        ))?;
        let mut config = SchedulerConfig::default();
        config.step_budget_ms = 25.0;
        let mut scheduler = Scheduler::new(1, config, cost_model()?)?;
        let plan = scheduler.schedule(&slots, 0);
        assert_eq!(plan.prefill_tokens, 256);
        assert!(plan.predicted_ms <= 25.0);
        Ok(())
    }

    #[test]
    fn deadline_budget_can_reduce_prefill_and_surfaces_single_request_limiter() -> Result<()> {
        let mut slots = RequestSlots::new(2, 1024)?;
        let decode = slots.acquire().expect("decode slot");
        slots.get_mut(decode)?.initialize(RequestInit::new(
            1,
            &[1; 128],
            512,
            0,
            400_000,
            50_000,
            32,
        ))?;
        slots.get_mut(decode)?.phase = RequestPhase::Decoding;
        slots.get_mut(decode)?.next_token_deadline_us = 18_000;
        let prefill = slots.acquire().expect("prefill slot");
        slots.get_mut(prefill)?.initialize(RequestInit::new(
            2,
            &[2; 512],
            640,
            0,
            400_000,
            50_000,
            40,
        ))?;

        let mut scheduler = Scheduler::new(2, SchedulerConfig::default(), cost_model()?)?;
        let plan = scheduler.schedule(&slots, 0);
        assert!(plan.deadline_limited);
        assert!(plan.deadline_limited_by_single_request());
        assert_eq!(plan.deadline_limiter_count, 1);
        assert!(plan.effective_budget_ms <= 16.0);
        assert!(plan.prefill_tokens <= 96);
        Ok(())
    }

    #[test]
    fn radix_prefilled_tokens_reduce_effective_prompt_work() -> Result<()> {
        let mut slots = RequestSlots::new(1, 1024)?;
        let slot = slots.acquire().expect("prefill slot");
        slots.get_mut(slot)?.initialize(RequestInit::new(
            1,
            &[2; 512],
            640,
            0,
            400_000,
            50_000,
            40,
        ))?;
        slots.get_mut(slot)?.prefilled = 384;
        let mut scheduler = Scheduler::new(1, SchedulerConfig::default(), cost_model()?)?;
        let plan = scheduler.schedule(&slots, 0);
        assert_eq!(plan.prefill_tokens, 128);
        assert!(matches!(
            plan.work(),
            [ScheduledWork::Prefill { tokens: 128, .. }]
        ));
        Ok(())
    }

    #[test]
    fn large_long_context_tail_burst_is_decode_only() -> Result<()> {
        let request_count = DECODE_TAIL_QUEUE_THRESHOLD;
        let mut slots = RequestSlots::new(request_count, 4096)?;
        for request_id in 0..request_count {
            let slot = slots.acquire().expect("tail slot");
            slots.get_mut(slot)?.initialize(RequestInit::new(
                request_id as u64,
                &[2; 2048],
                2176,
                0,
                400_000,
                50_000,
                136,
            ))?;
            slots.get_mut(slot)?.prefilled = 2032;
        }
        let mut scheduler =
            Scheduler::new(request_count, SchedulerConfig::default(), cost_model()?)?;
        let plan = scheduler.schedule(&slots, 10_000);
        let tail_prefills = plan
            .work()
            .iter()
            .filter_map(|work| match work {
                ScheduledWork::Prefill { tokens, .. } => Some(*tokens),
                ScheduledWork::Decode { .. } => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(tail_prefills.len(), request_count);
        assert!(tail_prefills.iter().all(|tokens| *tokens == 1));
        assert_eq!(plan.prefill_tokens, request_count);
        assert!(plan.predicted_ms < SchedulerConfig::default().step_budget_ms);
        Ok(())
    }

    #[test]
    fn long_tail_pressure_forces_short_cached_tails_to_singletons_too() -> Result<()> {
        let request_count = DECODE_TAIL_QUEUE_THRESHOLD;
        let long_count = 4usize;
        let mut slots = RequestSlots::new(request_count, 4096)?;
        for request_id in 0..request_count {
            let slot = slots.acquire().expect("mixed tail slot");
            if request_id < long_count {
                slots.get_mut(slot)?.initialize(RequestInit::new(
                    request_id as u64,
                    &[2; 2048],
                    2176,
                    0,
                    400_000,
                    50_000,
                    136,
                ))?;
                slots.get_mut(slot)?.prefilled = 2032;
            } else {
                slots.get_mut(slot)?.initialize(RequestInit::new(
                    request_id as u64,
                    &[2; 128],
                    256,
                    0,
                    400_000,
                    50_000,
                    16,
                ))?;
                slots.get_mut(slot)?.prefilled = 112;
            }
        }
        let mut scheduler =
            Scheduler::new(request_count, SchedulerConfig::default(), cost_model()?)?;
        let plan = scheduler.schedule(&slots, 10_000);
        let prefills = plan
            .work()
            .iter()
            .filter_map(|work| match work {
                ScheduledWork::Prefill { tokens, .. } => Some(*tokens),
                ScheduledWork::Decode { .. } => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(prefills.len(), request_count);
        assert!(prefills.iter().all(|tokens| *tokens == 1));
        Ok(())
    }

    #[test]
    fn small_long_context_tail_batch_keeps_chunked_prefill() -> Result<()> {
        let request_count = 8usize;
        let mut slots = RequestSlots::new(request_count, 4096)?;
        for request_id in 0..request_count {
            let slot = slots.acquire().expect("tail slot");
            slots.get_mut(slot)?.initialize(RequestInit::new(
                request_id as u64,
                &[2; 2048],
                2176,
                0,
                400_000,
                50_000,
                136,
            ))?;
            slots.get_mut(slot)?.prefilled = 2032;
        }
        let mut scheduler =
            Scheduler::new(request_count, SchedulerConfig::default(), cost_model()?)?;
        let plan = scheduler.schedule(&slots, 10_000);
        let tail_prefills = plan
            .work()
            .iter()
            .filter_map(|work| match work {
                ScheduledWork::Prefill { tokens, .. } => Some(*tokens),
                ScheduledWork::Decode { .. } => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(tail_prefills.len(), request_count);
        assert!(tail_prefills.iter().all(|tokens| *tokens == 16));
        Ok(())
    }
}