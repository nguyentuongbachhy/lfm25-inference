use anyhow::{Result, ensure};

use super::{HardwareCostModel, RequestPhase, RequestSlotId, RequestSlots};

const DECODE_TAIL_CONTEXT_THRESHOLD: usize = 1536;
const DECODE_TAIL_QUEUE_THRESHOLD: usize = 24;

#[derive(Debug, Clone, Copy)]
pub struct SchedulerConfig {
    pub step_budget_ms: f64,
    pub ttft_slo_us: u64,
    pub tpot_slo_us: u64,
    pub fp8_decode: bool,
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
}

impl BatchPlan {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            work: Vec::with_capacity(capacity),
            predicted_ms: 0.0,
            decode_count: 0,
            prefill_tokens: 0,
        }
    }

    pub fn work(&self) -> &[ScheduledWork] {
        &self.work
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
        self.prefill_order.clear();

        let mut maximum_context = 1usize;
        for (index, request) in requests.entries().iter().enumerate() {
            if request.phase == RequestPhase::Decoding {
                self.plan.work.push(ScheduledWork::Decode {
                    slot: RequestSlotId(index as u32),
                });
                maximum_context = maximum_context.max(request.tokens().len());
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

        let remaining_budget = (self.config.step_budget_ms - decode_ms).max(0.0);
        if remaining_budget <= 0.0 || self.config.maximum_prefill_tokens == 0 {
            return &self.plan;
        }

        for (index, request) in requests.entries().iter().enumerate() {
            if request.phase != RequestPhase::QueuedPrefill
                || request.prefilled >= request.prompt_len
            {
                continue;
            }
            let urgency = request
                .first_token_deadline_us
                .saturating_sub(now_us)
                .saturating_sub(now_us.saturating_sub(request.arrival_us) / 4);
            self.prefill_order
                .push((urgency, RequestSlotId(index as u32)));
        }
        self.prefill_order.sort_unstable_by_key(|entry| entry.0);
        let queued_prefills = self.prefill_order.len();

        for &(_, slot) in &self.prefill_order {
            if self.plan.prefill_tokens >= self.config.maximum_prefill_tokens {
                break;
            }
            let Ok(request) = requests.get(slot) else {
                continue;
            };
            let raw_remaining = request.prompt_len.saturating_sub(request.prefilled);
            if raw_remaining == 0 {
                continue;
            }

            // A large burst of page-aligned prefix hits can leave one final KV
            // page per request. Packing all of those tails as a single ragged
            // prefill makes attention see hundreds of query tokens at a long
            // context and badly underestimates the real cost. Draining one
            // token/request keeps every segment length at one, which lets the
            // model route the batch through the optimized paged-decode/MoK
            // path while preserving the exact sequential Conv/KV semantics.
            let decode_tail = queued_prefills >= DECODE_TAIL_QUEUE_THRESHOLD
                && request.prefilled >= DECODE_TAIL_CONTEXT_THRESHOLD
                && raw_remaining <= self.cost.page_size;

            let remaining = if decode_tail {
                1
            } else {
                raw_remaining.min(
                    self.config
                        .maximum_prefill_tokens
                        .saturating_sub(self.plan.prefill_tokens),
                )
            };
            if remaining == 0 {
                continue;
            }

            let mut chunk = 0usize;
            if decode_tail {
                let total_prefill = self.plan.prefill_tokens.saturating_add(1);
                if total_prefill <= self.config.maximum_prefill_tokens
                    && self.cost.predict_prefill_ms(total_prefill) <= remaining_budget
                {
                    chunk = 1;
                }
            } else {
                for candidate in [512usize, 256, 128, 64, 32, 16, 8, 4, 2, 1] {
                    if candidate > remaining {
                        continue;
                    }
                    let total_prefill = self.plan.prefill_tokens.saturating_add(candidate);
                    if self.cost.predict_prefill_ms(total_prefill) <= remaining_budget {
                        chunk = candidate;
                        break;
                    }
                }
            }
            if chunk == 0 {
                break;
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
        Ok(HardwareCostModel {
            schema_version: 1,
            gpu_name: "test".into(),
            page_size: 16,
            decode_bf16: CostCurve::new(vec![CostPoint {
                batch: 1,
                tokens: 1,
                context: 128,
                milliseconds: 8.0,
            }])?,
            decode_fp8: CostCurve::new(vec![CostPoint {
                batch: 1,
                tokens: 1,
                context: 128,
                milliseconds: 6.0,
            }])?,
            prefill_bf16: CostCurve::new(vec![CostPoint {
                batch: 1,
                tokens: 128,
                context: 128,
                milliseconds: 10.0,
            }])?,
            interactive_prompt_limit: 2048,
            ttft_slo_ms: 400.0,
            tpot_slo_ms: 50.0,
        })
    }

    #[test]
    fn decode_is_scheduled_before_aged_prefill_without_reallocation() -> Result<()> {
        let mut slots = RequestSlots::new(3, 1024)?;
        let decode = slots.acquire().expect("decode slot");
        slots
            .get_mut(decode)?
            .initialize(RequestInit::new(1, &[1], 512, 0, 400_000, 50_000, 32))?;
        slots.get_mut(decode)?.phase = RequestPhase::Decoding;
        let prefill = slots.acquire().expect("prefill slot");
        slots.get_mut(prefill)?.initialize(RequestInit::new(
            2,
            &[2; 256],
            512,
            0,
            400_000,
            50_000,
            32,
        ))?;
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
                request_id,
                &[2; 32],
                128,
                0,
                400_000,
                50_000,
                8,
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
    fn large_long_context_tail_burst_is_drained_one_token_per_request() -> Result<()> {
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
        let mut scheduler = Scheduler::new(
            request_count,
            SchedulerConfig::default(),
            cost_model()?,
        )?;
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
        let mut scheduler = Scheduler::new(
            request_count,
            SchedulerConfig::default(),
            cost_model()?,
        )?;
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
