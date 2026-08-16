use anyhow::{Result, ensure};

use super::{HardwareCostModel, RequestPhase, RequestSlotId, RequestSlots};

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
        })
    }

    pub fn schedule(&mut self, requests: &RequestSlots, now_us: u64) -> &BatchPlan {
        self.plan.work.clear();
        self.plan.predicted_ms = 0.0;
        self.plan.decode_count = 0;
        self.plan.prefill_tokens = 0;

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
        self.plan.predicted_ms = if self.plan.decode_count == 0 {
            0.0
        } else {
            self.cost.predict_decode_ms(
                self.plan.decode_count,
                maximum_context,
                self.config.fp8_decode,
            )
        };

        let remaining_budget = (self.config.step_budget_ms - self.plan.predicted_ms).max(0.0);
        let mut best_prefill = None;
        let mut best_urgency = u64::MAX;
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
            if urgency < best_urgency {
                best_urgency = urgency;
                best_prefill = Some((RequestSlotId(index as u32), request));
            }
        }
        if let Some((slot, request)) = best_prefill {
            let remaining = request
                .prompt_len
                .saturating_sub(request.prefilled)
                .min(self.config.maximum_prefill_tokens);
            let chunk = self.cost.largest_prefill_chunk(remaining, remaining_budget);
            if chunk > 0 {
                self.plan.work.push(ScheduledWork::Prefill {
                    slot,
                    tokens: chunk,
                });
                self.plan.prefill_tokens = chunk;
                self.plan.predicted_ms += self.cost.predict_prefill_ms(chunk);
            }
        }
        &self.plan
    }
}

#[cfg(test)]
mod tests {
    use crate::scheduler::{CostCurve, CostPoint};

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
            .initialize(1, &[1], 512, 0, 400_000, 50_000, 32)?;
        slots.get_mut(decode)?.phase = RequestPhase::Decoding;
        let prefill = slots.acquire().expect("prefill slot");
        slots
            .get_mut(prefill)?
            .initialize(2, &[2; 256], 512, 0, 400_000, 50_000, 32)?;
        let mut scheduler = Scheduler::new(3, SchedulerConfig::default(), cost_model()?)?;
        let capacity = scheduler.plan.work.capacity();
        let plan = scheduler.schedule(&slots, 10_000);
        assert!(matches!(plan.work()[0], ScheduledWork::Decode { .. }));
        assert!(plan.prefill_tokens > 0);
        assert_eq!(scheduler.plan.work.capacity(), capacity);
        Ok(())
    }
}
