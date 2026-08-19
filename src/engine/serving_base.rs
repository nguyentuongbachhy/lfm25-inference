use std::{
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result, ensure};
use half::bf16;
use serde::Serialize;
use tokio::sync::{mpsc, oneshot};

use crate::{
    generation::SamplingConfig,
    model::RaggedBatchInput,
    ops,
    scheduler::{
        HardwareCostModel, RequestInit, RequestPhase, RequestSlotId, RequestSlots, ScheduledWork,
        Scheduler, SchedulerConfig,
    },
};

use super::{Engine, GenerationMetrics};

#[derive(Debug, Clone)]
pub struct ContinuousEngineConfig {
    pub maximum_request_slots: usize,
    pub maximum_sequence_tokens: usize,
    pub maximum_batch_tokens: usize,
    pub physical_kv_pages: usize,
    pub queue_capacity: usize,
    pub scheduler: SchedulerConfig,
    pub cost_model: HardwareCostModel,
    pub trace_steps: bool,
}

impl ContinuousEngineConfig {
    pub fn from_cost_model(cost_model: HardwareCostModel) -> Result<Self> {
        ensure!(
            cost_model.schema_version == 1,
            "unsupported hardware profile schema {}",
            cost_model.schema_version
        );
        Ok(Self {
            maximum_request_slots: 64,
            maximum_sequence_tokens: 32_768,
            maximum_batch_tokens: 64 + 512,
            physical_kv_pages: 1,
            queue_capacity: 128,
            scheduler: SchedulerConfig::default(),
            cost_model,
            trace_steps: false,
        })
    }
}

pub struct PreparedRequest {
    pub request_id: u64,
    pub token_ids: Vec<u32>,
    pub maximum_new_tokens: usize,
    pub stop_on_eos: bool,
    pub sampling: SamplingConfig,
    pub arrived: Instant,
    pub response: oneshot::Sender<Result<ServingCompletion, ServingError>>,
}

#[derive(Debug)]
pub struct ServingError {
    pub status: &'static str,
    pub message: String,
}

impl ServingError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: "400 Bad Request",
            message: message.into(),
        }
    }

    fn overloaded(message: impl Into<String>) -> Self {
        Self {
            status: "503 Service Unavailable",
            message: message.into(),
        }
    }
}

#[derive(Debug)]
pub struct ServingCompletion {
    pub token_ids: Vec<u32>,
    pub prompt_tokens: usize,
    pub finish_reason: &'static str,
    pub metrics: GenerationMetrics,
}

#[derive(Debug, Serialize)]
pub struct ServingOwnerReport {
    pub wall_ms: f64,
    pub engine_steps: u64,
    pub completed_requests: u64,
    pub generated_tokens: u64,
    pub model_input_tokens: u64,
    pub maximum_active_requests: usize,
    pub transfers: crate::scheduler::TransferCounters,
    pub kv_total_pages: usize,
    pub kv_peak_allocated_pages: usize,
    pub bf16_pool_hits: u64,
    pub bf16_pool_misses: u64,
    pub bf16_pool_available_elements: usize,
    pub bf16_pool_dropped_elements: u64,
    pub fp8_pool_hits: u64,
    pub fp8_pool_misses: u64,
    pub fp8_pool_available_elements: usize,
    pub fp8_pool_dropped_elements: u64,
    pub free_vram_bytes_before_cache_drop: usize,
    pub total_vram_bytes: usize,
    pub resident_weight_bytes: usize,
    pub kv_arena_bytes: usize,
    pub temp_pool_retained_bytes: usize,
    pub cublaslt_workspace_bytes: usize,
}

#[derive(Clone)]
pub struct ServingHandle {
    sender: mpsc::Sender<PreparedRequest>,
}

impl ServingHandle {
    pub fn try_submit(&self, request: PreparedRequest) -> Result<(), PreparedRequest> {
        self.sender
            .try_send(request)
            .map_err(|error| error.into_inner())
    }

    pub fn channel(capacity: usize) -> (Self, mpsc::Receiver<PreparedRequest>) {
        let (sender, receiver) = mpsc::channel(capacity);
        (Self { sender }, receiver)
    }
}

struct ResponseState {
    response: Option<oneshot::Sender<Result<ServingCompletion, ServingError>>>,
    arrived: Instant,
    first_token_ready: Option<Instant>,
    last_token_ready: Option<Instant>,
    inter_token_ms: Vec<f64>,
    prompt_tokens: usize,
    maximum_new_tokens: usize,
    generated_tokens: usize,
    stop_on_eos: bool,
    first_scheduled: Option<Instant>,
    prefill_step_ms: f64,
    decode_step_ms: f64,
    prefill_submit_cpu_ms: f64,
    prefill_d2h_ms: f64,
    decode_submit_cpu_ms: f64,
    decode_d2h_ms: f64,
    bf16_pool_hits: u64,
    bf16_pool_misses: u64,
    fp8_pool_hits: u64,
    fp8_pool_misses: u64,
    scheduler_cpu_ms: f64,
}

impl ResponseState {
    fn vacant(maximum_tokens: usize) -> Self {
        Self {
            response: None,
            arrived: Instant::now(),
            first_token_ready: None,
            last_token_ready: None,
            inter_token_ms: Vec::with_capacity(maximum_tokens),
            prompt_tokens: 0,
            maximum_new_tokens: 0,
            generated_tokens: 0,
            stop_on_eos: true,
            first_scheduled: None,
            prefill_step_ms: 0.0,
            decode_step_ms: 0.0,
            prefill_submit_cpu_ms: 0.0,
            prefill_d2h_ms: 0.0,
            decode_submit_cpu_ms: 0.0,
            decode_d2h_ms: 0.0,
            bf16_pool_hits: 0,
            bf16_pool_misses: 0,
            fp8_pool_hits: 0,
            fp8_pool_misses: 0,
            scheduler_cpu_ms: 0.0,
        }
    }
}

impl Engine {
    pub fn tokenizer_clone(&self) -> crate::tokenizer::Lfm2Tokenizer {
        self.tokenizer.clone()
    }

    pub fn run_continuous_owner(
        self,
        config: ContinuousEngineConfig,
        receiver: mpsc::Receiver<PreparedRequest>,
        ready: std::sync::mpsc::SyncSender<()>,
    ) -> Result<ServingOwnerReport> {
        ensure!(
            config.maximum_request_slots > 0,
            "continuous engine needs request slots"
        );
        ensure!(
            config
                .maximum_request_slots
                .checked_add(config.scheduler.maximum_prefill_tokens)
                .is_some_and(|required| required <= config.maximum_batch_tokens),
            "maximum batch tokens cannot cover decode slots plus prefill chunk"
        );
        run_owner(self, config, receiver, ready)
    }

    pub fn continuous_config(
        &self,
        cost_model: HardwareCostModel,
    ) -> Result<ContinuousEngineConfig> {
        let actual_gpu = self.runtime.device_name()?;
        ensure!(
            actual_gpu == cost_model.gpu_name,
            "hardware profile targets {:?}, active GPU is {:?}",
            cost_model.gpu_name,
            actual_gpu
        );
        ensure!(
            cost_model.page_size == self.config.kv_page_size.value(),
            "hardware profile page size {} does not match engine page size {}",
            cost_model.page_size,
            self.config.kv_page_size.value(),
        );
        let mut config = ContinuousEngineConfig::from_cost_model(cost_model)?;
        config.maximum_sequence_tokens = config
            .maximum_sequence_tokens
            .min(self.model.config().max_position_embeddings);
        let (free_bytes, total_bytes) = self.runtime.memory_info()?;
        let safety_bytes = (total_bytes / 10).max(512 * 1024 * 1024);
        let temporary_pool_budget =
            (64usize * 1024 * 1024 * std::mem::size_of::<bf16>()).saturating_add(32 * 1024 * 1024);
        let available_for_kv = free_bytes
            .saturating_sub(safety_bytes)
            .saturating_sub(temporary_pool_budget);
        let attention_layers = self
            .model
            .config()
            .layer_types
            .iter()
            .filter(|kind| kind.as_str() == "full_attention")
            .count();
        let bytes_per_page = attention_layers
            .checked_mul(2)
            .and_then(|value| value.checked_mul(self.model.config().num_key_value_heads))
            .and_then(|value| value.checked_mul(self.config.kv_page_size.value()))
            .and_then(|value| value.checked_mul(self.model.config().head_dim()))
            .and_then(|value| value.checked_mul(std::mem::size_of::<bf16>()))
            .context("KV page byte size overflow")?;
        let maximum_useful_pages = config
            .maximum_request_slots
            .checked_mul(
                config
                    .maximum_sequence_tokens
                    .div_ceil(self.config.kv_page_size.value()),
            )
            .context("maximum KV page count overflow")?;
        config.physical_kv_pages = (available_for_kv / bytes_per_page)
            .min(maximum_useful_pages)
            .max(1);
        Ok(config)
    }
}

fn run_owner(
    engine: Engine,
    config: ContinuousEngineConfig,
    mut receiver: mpsc::Receiver<PreparedRequest>,
    ready: std::sync::mpsc::SyncSender<()>,
) -> Result<ServingOwnerReport> {
    let mut cache = engine.model.new_batch_cache(
        &engine.runtime,
        config.maximum_request_slots,
        config.maximum_batch_tokens,
        config.physical_kv_pages,
        engine.config.kv_page_size,
    )?;
    let mut slots =
        RequestSlots::new(config.maximum_request_slots, config.maximum_sequence_tokens)?;
    let mut responses: Vec<ResponseState> = (0..config.maximum_request_slots)
        .map(|_| ResponseState::vacant(config.maximum_sequence_tokens))
        .collect();
    let mut scheduler = Scheduler::new(
        config.maximum_request_slots,
        config.scheduler,
        config.cost_model.clone(),
    )?;
    let mut work = Vec::with_capacity(config.maximum_request_slots + 1);
    let mut token_ids = Vec::with_capacity(config.maximum_batch_tokens);
    let mut positions = Vec::with_capacity(config.maximum_batch_tokens);
    let mut request_slots = Vec::with_capacity(config.maximum_batch_tokens);
    let mut segment_offsets = Vec::with_capacity(config.maximum_request_slots + 2);
    let mut segment_slots = Vec::with_capacity(config.maximum_request_slots + 1);
    let mut output_rows = Vec::with_capacity(config.maximum_request_slots + 1);
    let mut finished = Vec::with_capacity(config.maximum_request_slots);
    let mut sampled_host = engine
        .runtime
        .pinned_u32(config.maximum_request_slots + 1)?;
    warm_serving_path(&engine, &config, &mut cache)?;
    cache.begin_serving_measurement();
    let serving_bf16_pool_started = engine.runtime.bf16_pool_stats();
    let serving_fp8_pool_started = engine.runtime.fp8_pool_stats();
    ready
        .send(())
        .map_err(|_| anyhow::anyhow!("async frontend stopped before GPU owner became ready"))?;
    let owner_started = Instant::now();
    let mut engine_steps = 0u64;
    let mut completed_requests = 0u64;
    let mut generated_tokens = 0u64;
    let mut model_input_tokens = 0u64;
    let mut maximum_active_requests = 0usize;

    loop {
        while let Ok(request) = receiver.try_recv() {
            admit_request(
                &engine,
                &config,
                &mut cache,
                &mut slots,
                &mut responses,
                request,
            )?;
        }
        if slots.free_count() == config.maximum_request_slots {
            let Some(request) = receiver.blocking_recv() else {
                break;
            };
            admit_request(
                &engine,
                &config,
                &mut cache,
                &mut slots,
                &mut responses,
                request,
            )?;
            continue;
        }
        maximum_active_requests = maximum_active_requests.max(
            config
                .maximum_request_slots
                .saturating_sub(slots.free_count()),
        );

        let scheduler_started = Instant::now();
        let now_us = monotonic_us();
        work.clear();
        work.extend_from_slice(scheduler.schedule(&slots, now_us).work());
        if work.is_empty() {
            thread::sleep(Duration::from_micros(100));
            continue;
        }
        engine_steps = engine_steps.saturating_add(1);
        let scheduler_elapsed_ms = scheduler_started.elapsed().as_secs_f64() * 1000.0;
        token_ids.clear();
        positions.clear();
        request_slots.clear();
        segment_offsets.clear();
        segment_slots.clear();
        output_rows.clear();
        segment_offsets.push(0);
        for item in &work {
            let (slot_id, start, length) = match *item {
                ScheduledWork::Decode { slot } => {
                    let request = slots.get(slot)?;
                    (slot, request.tokens().len() - 1, 1)
                }
                ScheduledWork::Prefill { slot, tokens } => {
                    let request = slots.get(slot)?;
                    let remaining = request.prompt_len.saturating_sub(request.prefilled);
                    (slot, request.prefilled, tokens.min(remaining))
                }
            };
            let request = slots.get(slot_id)?;
            let end = start
                .checked_add(length)
                .context("scheduled token range overflow")?;
            ensure!(
                end <= request.tokens().len(),
                "scheduled token range exceeds request"
            );
            token_ids.extend_from_slice(&request.tokens()[start..end]);
            for position in start..end {
                positions.push(u32::try_from(position)?);
            }
            request_slots.extend(std::iter::repeat_n(slot_id.0, length));
            segment_offsets.push(u32::try_from(token_ids.len())?);
            segment_slots.push(slot_id.0);
            output_rows.push(u32::try_from(token_ids.len() - 1)?);
        }
        model_input_tokens = model_input_tokens.saturating_add(token_ids.len() as u64);
        let step_started = Instant::now();
        for item in &work {
            let slot = match *item {
                ScheduledWork::Decode { slot } | ScheduledWork::Prefill { slot, .. } => slot,
            };
            let state = &mut responses[slot.0 as usize];
            state.scheduler_cpu_ms += scheduler_elapsed_ms;
            if state.first_scheduled.is_none() {
                state.first_scheduled = Some(step_started);
            }
        }
        let bf16_pool_started = engine.runtime.bf16_pool_stats();
        let fp8_pool_started = engine.runtime.fp8_pool_stats();
        let gpu_started = engine.runtime.record_timing_event()?;
        let submit_started = Instant::now();
        let logits = engine.model.forward_ragged_batch(
            &engine.runtime,
            &mut cache,
            RaggedBatchInput {
                token_ids: &token_ids,
                positions: &positions,
                request_slots: &request_slots,
                segment_offsets: &segment_offsets,
                segment_slots: &segment_slots,
                output_rows: &output_rows,
            },
        )?;
        let sampled = ops::argmax_rows_bf16(&engine.runtime, &logits)?;
        let gpu_finished = engine.runtime.record_timing_event()?;
        let submit_cpu_ms = submit_started.elapsed().as_secs_f64() * 1000.0;
        let gpu_ms = engine.runtime.elapsed_ms(&gpu_started, &gpu_finished)?;
        let download_started = Instant::now();
        engine
            .runtime
            .download_u32_into(&sampled, &mut sampled_host)?;
        let sampled = sampled_host
            .as_slice()
            .context("failed to synchronize pinned token output")?;
        let d2h_ms = download_started.elapsed().as_secs_f64() * 1000.0;
        let bf16_pool_finished = engine.runtime.bf16_pool_stats();
        let fp8_pool_finished = engine.runtime.fp8_pool_stats();
        let bf16_hits = bf16_pool_finished
            .hits
            .saturating_sub(bf16_pool_started.hits);
        let bf16_misses = bf16_pool_finished
            .misses
            .saturating_sub(bf16_pool_started.misses);
        let fp8_hits = fp8_pool_finished.hits.saturating_sub(fp8_pool_started.hits);
        let fp8_misses = fp8_pool_finished
            .misses
            .saturating_sub(fp8_pool_started.misses);
        let ready = Instant::now();
        let step_ms = ready.duration_since(step_started).as_secs_f64() * 1000.0;
        if config.trace_steps
            && work
                .iter()
                .any(|item| matches!(item, ScheduledWork::Prefill { .. }))
        {
            eprintln!(
                "serving_step tokens={} segments={} wall_ms={step_ms:.3} submit_cpu_ms={submit_cpu_ms:.3} gpu_ms={gpu_ms:.3} d2h_ms={d2h_ms:.3} bf16_hits={bf16_hits} bf16_misses={bf16_misses} bf16_retained={} bf16_dropped={}",
                token_ids.len(),
                work.len(),
                bf16_pool_finished.available_elements,
                bf16_pool_finished.dropped_elements,
            );
        }
        finished.clear();
        for (row, item) in work.iter().enumerate() {
            let slot = match *item {
                ScheduledWork::Decode { slot } | ScheduledWork::Prefill { slot, .. } => slot,
            };
            let request = slots.get_mut(slot)?;
            if let ScheduledWork::Prefill { tokens, .. } = *item {
                let state = &mut responses[slot.0 as usize];
                state.prefill_step_ms += gpu_ms;
                state.prefill_submit_cpu_ms += submit_cpu_ms;
                state.prefill_d2h_ms += d2h_ms;
                state.bf16_pool_hits = state.bf16_pool_hits.saturating_add(bf16_hits);
                state.bf16_pool_misses = state.bf16_pool_misses.saturating_add(bf16_misses);
                state.fp8_pool_hits = state.fp8_pool_hits.saturating_add(fp8_hits);
                state.fp8_pool_misses = state.fp8_pool_misses.saturating_add(fp8_misses);
                request.prefilled = request
                    .prefilled
                    .saturating_add(tokens)
                    .min(request.prompt_len);
                if request.prefilled < request.prompt_len {
                    continue;
                }
            }
            if matches!(*item, ScheduledWork::Decode { .. }) {
                let state = &mut responses[slot.0 as usize];
                state.decode_step_ms += gpu_ms;
                state.decode_submit_cpu_ms += submit_cpu_ms;
                state.decode_d2h_ms += d2h_ms;
                state.bf16_pool_hits = state.bf16_pool_hits.saturating_add(bf16_hits);
                state.bf16_pool_misses = state.bf16_pool_misses.saturating_add(bf16_misses);
                state.fp8_pool_hits = state.fp8_pool_hits.saturating_add(fp8_hits);
                state.fp8_pool_misses = state.fp8_pool_misses.saturating_add(fp8_misses);
            }
            let token = sampled[row];
            generated_tokens = generated_tokens.saturating_add(1);
            let state = &mut responses[slot.0 as usize];
            if let Some(last) = state.last_token_ready {
                state
                    .inter_token_ms
                    .push(ready.duration_since(last).as_secs_f64() * 1000.0);
            } else {
                state.first_token_ready = Some(ready);
            }
            state.last_token_ready = Some(ready);
            state.generated_tokens += 1;
            let eos = state.stop_on_eos && token == engine.model.config().eos_token_id;
            let length = state.generated_tokens >= state.maximum_new_tokens;
            if !eos && !length {
                request.push_token(token, now_us, config.scheduler.tpot_slo_us)?;
            } else {
                if !eos {
                    request.push_token(token, now_us, config.scheduler.tpot_slo_us)?;
                }
                finished.push((slot, if eos { "stop" } else { "length" }));
            }
        }
        for (slot, reason) in finished.drain(..) {
            finish_request(
                &engine,
                &mut cache,
                &mut slots,
                &mut responses,
                slot,
                reason,
            )?;
            completed_requests = completed_requests.saturating_add(1);
        }
    }
    let transfers = cache.transfers();
    let kv = cache.kv_snapshot();
    let bf16 = engine.runtime.bf16_pool_stats();
    let fp8 = engine.runtime.fp8_pool_stats();
    let (free_vram_bytes_before_cache_drop, total_vram_bytes) = engine.runtime.memory_info()?;
    let attention_layers = engine
        .model
        .config()
        .layer_types
        .iter()
        .filter(|kind| kind.as_str() == "full_attention")
        .count();
    let kv_arena_bytes = kv
        .total_pages
        .saturating_mul(attention_layers)
        .saturating_mul(2)
        .saturating_mul(engine.model.config().num_key_value_heads)
        .saturating_mul(engine.config.kv_page_size.value())
        .saturating_mul(engine.model.config().head_dim())
        .saturating_mul(std::mem::size_of::<bf16>());
    Ok(ServingOwnerReport {
        wall_ms: owner_started.elapsed().as_secs_f64() * 1000.0,
        engine_steps,
        completed_requests,
        generated_tokens,
        model_input_tokens,
        maximum_active_requests,
        transfers,
        kv_total_pages: kv.total_pages,
        kv_peak_allocated_pages: kv.peak_allocated_pages,
        bf16_pool_hits: bf16.hits.saturating_sub(serving_bf16_pool_started.hits),
        bf16_pool_misses: bf16.misses.saturating_sub(serving_bf16_pool_started.misses),
        bf16_pool_available_elements: bf16.available_elements,
        bf16_pool_dropped_elements: bf16
            .dropped_elements
            .saturating_sub(serving_bf16_pool_started.dropped_elements),
        fp8_pool_hits: fp8.hits.saturating_sub(serving_fp8_pool_started.hits),
        fp8_pool_misses: fp8.misses.saturating_sub(serving_fp8_pool_started.misses),
        fp8_pool_available_elements: fp8.available_elements,
        fp8_pool_dropped_elements: fp8
            .dropped_elements
            .saturating_sub(serving_fp8_pool_started.dropped_elements),
        free_vram_bytes_before_cache_drop,
        total_vram_bytes,
        resident_weight_bytes: engine.model.resident_weight_bytes(),
        kv_arena_bytes,
        temp_pool_retained_bytes: bf16
            .available_elements
            .saturating_mul(std::mem::size_of::<bf16>())
            .saturating_add(fp8.available_elements),
        cublaslt_workspace_bytes: engine.runtime.blaslt().workspace_size(),
    })
}

fn warm_serving_path(
    engine: &Engine,
    config: &ContinuousEngineConfig,
    cache: &mut crate::model::BatchModelCache,
) -> Result<()> {
    let pool_started = engine.runtime.bf16_pool_stats();
    let fp8_pool_started = engine.runtime.fp8_pool_stats();
    let decode_slots = config.maximum_request_slots.saturating_sub(1);
    let prefill_tokens = config.scheduler.maximum_prefill_tokens;
    let total_tokens = decode_slots
        .checked_add(prefill_tokens)
        .context("serving warmup token count overflow")?;
    ensure!(
        total_tokens > 0 && total_tokens <= config.maximum_batch_tokens,
        "invalid serving warmup shape"
    );
    let mut token_ids = Vec::with_capacity(total_tokens);
    let mut positions = Vec::with_capacity(total_tokens);
    let mut request_slots = Vec::with_capacity(total_tokens);
    let mut segment_offsets = Vec::with_capacity(config.maximum_request_slots + 1);
    let mut segment_slots = Vec::with_capacity(config.maximum_request_slots);
    let mut output_rows = Vec::with_capacity(config.maximum_request_slots);
    segment_offsets.push(0u32);
    for slot in 0..decode_slots {
        cache.reserve(slot, 1)?;
        token_ids.push(engine.model.config().bos_token_id);
        positions.push(0u32);
        request_slots.push(u32::try_from(slot)?);
        segment_offsets.push(u32::try_from(token_ids.len())?);
        segment_slots.push(u32::try_from(slot)?);
        output_rows.push(u32::try_from(token_ids.len() - 1)?);
    }
    let prefill_slot = decode_slots;
    cache.reserve(prefill_slot, prefill_tokens)?;
    for position in 0..prefill_tokens {
        token_ids.push(if position == 0 {
            engine.model.config().bos_token_id
        } else {
            42
        });
        positions.push(u32::try_from(position)?);
        request_slots.push(u32::try_from(prefill_slot)?);
    }
    segment_offsets.push(u32::try_from(token_ids.len())?);
    segment_slots.push(u32::try_from(prefill_slot)?);
    output_rows.push(u32::try_from(token_ids.len() - 1)?);
    let logits = engine.model.forward_ragged_batch(
        &engine.runtime,
        cache,
        RaggedBatchInput {
            token_ids: &token_ids,
            positions: &positions,
            request_slots: &request_slots,
            segment_offsets: &segment_offsets,
            segment_slots: &segment_slots,
            output_rows: &output_rows,
        },
    )?;
    let sampled = ops::argmax_rows_bf16(&engine.runtime, &logits)?;
    engine.runtime.synchronize()?;
    drop(sampled);
    drop(logits);
    for slot in 0..config.maximum_request_slots {
        cache.release(&engine.runtime, slot)?;
    }
    if engine.model.decode_fp8_enabled() {
        cache.reserve(0, 1)?;
        let logits = engine.model.forward_decode_batch(
            &engine.runtime,
            cache,
            &[engine.model.config().bos_token_id],
            &[0u32],
            &[0u32],
        )?;
        let sampled = ops::argmax_rows_bf16(&engine.runtime, &logits)?;
        engine.runtime.synchronize()?;
        drop(sampled);
        drop(logits);
        cache.release(&engine.runtime, 0)?;
    }
    engine.runtime.synchronize()?;
    let pool_finished = engine.runtime.bf16_pool_stats();
    let fp8_pool_finished = engine.runtime.fp8_pool_stats();
    let dropped = pool_finished
        .dropped_elements
        .saturating_sub(pool_started.dropped_elements);
    ensure!(
        dropped == 0,
        "BF16 temp arena dropped {dropped} elements during serving warmup"
    );
    let fp8_dropped = fp8_pool_finished
        .dropped_elements
        .saturating_sub(fp8_pool_started.dropped_elements);
    ensure!(
        fp8_dropped == 0,
        "FP8 temp arena dropped {fp8_dropped} elements during serving warmup"
    );
    eprintln!(
        "serving warmup: tokens={} output_rows={} bf16_pool_elements={} bf16_pool_misses={} fp8_pool_elements={} fp8_pool_misses={}",
        total_tokens,
        output_rows.len(),
        pool_finished.available_elements,
        pool_finished.misses.saturating_sub(pool_started.misses),
        fp8_pool_finished.available_elements,
        fp8_pool_finished
            .misses
            .saturating_sub(fp8_pool_started.misses),
    );
    Ok(())
}

fn admit_request(
    engine: &Engine,
    config: &ContinuousEngineConfig,
    cache: &mut crate::model::BatchModelCache,
    slots: &mut RequestSlots,
    responses: &mut [ResponseState],
    request: PreparedRequest,
) -> Result<()> {
    if request.sampling.temperature != 0.0 || request.sampling.repetition_penalty != 1.0 {
        let _ = request.response.send(Err(ServingError::bad_request(
            "continuous path currently requires greedy sampling",
        )));
        return Ok(());
    }
    let total = request
        .token_ids
        .len()
        .checked_add(request.maximum_new_tokens)
        .context("request length overflow")?;
    if total > config.maximum_sequence_tokens
        || total > engine.model.config().max_position_embeddings
    {
        let _ = request.response.send(Err(ServingError::bad_request(
            "request exceeds configured sequence capacity",
        )));
        return Ok(());
    }
    if request.token_ids.len() > config.cost_model.interactive_prompt_limit {
        let _ = request.response.send(Err(ServingError::overloaded(format!(
            "prompt has {} tokens; interactive limit for this hardware profile is {} and no long-context queue is configured",
            request.token_ids.len(),
            config.cost_model.interactive_prompt_limit,
        ))));
        return Ok(());
    }
    let mut decode_count = 1usize;
    let mut maximum_context = total;
    let mut predicted_prefill_queue_ms = config
        .cost_model
        .predict_prefill_ms(request.token_ids.len());
    for state in slots.entries() {
        if state.phase == RequestPhase::Decoding {
            decode_count = decode_count.saturating_add(1);
            maximum_context = maximum_context.max(state.tokens().len());
        } else if state.phase == RequestPhase::QueuedPrefill {
            let remaining = state.prompt_len.saturating_sub(state.prefilled);
            predicted_prefill_queue_ms += config.cost_model.predict_prefill_ms(remaining);
        }
    }
    let predicted_decode_ms = config.cost_model.predict_decode_ms(
        decode_count,
        maximum_context,
        config.scheduler.fp8_decode,
    );
    if predicted_decode_ms >= config.scheduler.step_budget_ms
        || predicted_decode_ms * 1000.0 >= config.scheduler.tpot_slo_us as f64
    {
        let _ = request.response.send(Err(ServingError::overloaded(format!(
            "admission rejected: predicted decode p95 {predicted_decode_ms:.3} ms exceeds latency budget"
        ))));
        return Ok(());
    }
    let elapsed_ms = request.arrived.elapsed().as_secs_f64() * 1000.0;
    if elapsed_ms + predicted_prefill_queue_ms >= config.cost_model.ttft_slo_ms {
        let _ = request.response.send(Err(ServingError::overloaded(format!(
            "admission rejected: predicted TTFT {:.3} ms exceeds {:.3} ms SLO",
            elapsed_ms + predicted_prefill_queue_ms,
            config.cost_model.ttft_slo_ms,
        ))));
        return Ok(());
    }
    let Some(slot) = slots.acquire() else {
        let _ = request.response.send(Err(ServingError::overloaded(
            "engine request slots exhausted",
        )));
        return Ok(());
    };
    if let Err(error) = cache.reserve(slot.0 as usize, total) {
        slots.release(slot)?;
        let _ = request.response.send(Err(ServingError::overloaded(format!(
            "KV admission rejected: {error}"
        ))));
        return Ok(());
    }
    let reserved_pages = total.div_ceil(engine.config.kv_page_size.value());
    slots.get_mut(slot)?.initialize(RequestInit::new(
        request.request_id,
        &request.token_ids,
        total,
        monotonic_us(),
        config.scheduler.ttft_slo_us,
        config.scheduler.tpot_slo_us,
        reserved_pages,
    ))?;
    let state = &mut responses[slot.0 as usize];
    state.response = Some(request.response);
    state.arrived = request.arrived;
    state.first_token_ready = None;
    state.last_token_ready = None;
    state.inter_token_ms.clear();
    state.prompt_tokens = request.token_ids.len();
    state.maximum_new_tokens = request.maximum_new_tokens;
    state.generated_tokens = 0;
    state.stop_on_eos = request.stop_on_eos;
    state.first_scheduled = None;
    state.prefill_step_ms = 0.0;
    state.decode_step_ms = 0.0;
    state.prefill_submit_cpu_ms = 0.0;
    state.prefill_d2h_ms = 0.0;
    state.decode_submit_cpu_ms = 0.0;
    state.decode_d2h_ms = 0.0;
    state.bf16_pool_hits = 0;
    state.bf16_pool_misses = 0;
    state.fp8_pool_hits = 0;
    state.fp8_pool_misses = 0;
    state.scheduler_cpu_ms = 0.0;
    Ok(())
}

fn finish_request(
    engine: &Engine,
    cache: &mut crate::model::BatchModelCache,
    slots: &mut RequestSlots,
    responses: &mut [ResponseState],
    slot: RequestSlotId,
    finish_reason: &'static str,
) -> Result<()> {
    let request = slots.get(slot)?;
    let state = &mut responses[slot.0 as usize];
    let generated_start = state.prompt_tokens;
    let generated = request.tokens()[generated_start..].to_vec();
    let ttft_ms = state
        .first_token_ready
        .map(|time| time.duration_since(state.arrived).as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let tpot_mean = if state.inter_token_ms.is_empty() {
        None
    } else {
        Some(state.inter_token_ms.iter().sum::<f64>() / state.inter_token_ms.len() as f64)
    };
    let total_ms = state
        .last_token_ready
        .map(|time| time.duration_since(state.arrived).as_secs_f64() * 1000.0)
        .unwrap_or(ttft_ms);
    let metrics = GenerationMetrics {
        tokenization_ms: 0.0,
        queue_delay_ms: state
            .first_scheduled
            .map(|time| time.duration_since(state.arrived).as_secs_f64() * 1000.0)
            .unwrap_or(0.0),
        scheduler_cpu_ms: state.scheduler_cpu_ms,
        cache_allocation_cpu_ms: 0.0,
        cache_initialization_gpu_ms: 0.0,
        prefill_gpu_ms: state.prefill_step_ms,
        prefill_submit_cpu_ms: state.prefill_submit_cpu_ms,
        prefill_d2h_ms: state.prefill_d2h_ms,
        first_token_gpu_wait_and_sampling_ms: 0.0,
        ttft_ms,
        decode_gpu_ms: state.decode_step_ms,
        decode_submit_cpu_ms: state.decode_submit_cpu_ms,
        decode_d2h_ms: state.decode_d2h_ms,
        decode_total_ms: (total_ms - ttft_ms).max(0.0),
        tpot_mean_ms: tpot_mean,
        tpot_p50_ms: percentile(&state.inter_token_ms, 0.50),
        tpot_p95_ms: percentile(&state.inter_token_ms, 0.95),
        decode_tokens_per_second: tpot_mean.map(|value| 1000.0 / value),
        gpu_wait_and_sampling_total_ms: 0.0,
        bf16_pool_hits: state.bf16_pool_hits,
        bf16_pool_misses: state.bf16_pool_misses,
        fp8_pool_hits: state.fp8_pool_hits,
        fp8_pool_misses: state.fp8_pool_misses,
        decode_bf16_pool_hits: 0,
        decode_bf16_pool_misses: 0,
        decode_fp8_pool_hits: 0,
        decode_fp8_pool_misses: 0,
        bf16_pool_available_elements: engine.runtime.bf16_pool_stats().available_elements,
        bf16_pool_dropped_elements: engine.runtime.bf16_pool_stats().dropped_elements,
        fp8_pool_available_elements: engine.runtime.fp8_pool_stats().available_elements,
        fp8_pool_dropped_elements: engine.runtime.fp8_pool_stats().dropped_elements,
        bf16_pool_internal_fragment_elements: engine
            .runtime
            .bf16_pool_stats()
            .internal_fragment_elements,
        fp8_pool_internal_fragment_elements: engine
            .runtime
            .fp8_pool_stats()
            .internal_fragment_elements,
        detokenization_ms: 0.0,
        total_ms,
    };
    if let Some(sender) = state.response.take() {
        let _ = sender.send(Ok(ServingCompletion {
            token_ids: generated,
            prompt_tokens: state.prompt_tokens,
            finish_reason,
            metrics,
        }));
    }
    cache.release(&engine.runtime, slot.0 as usize)?;
    slots.release(slot)?;
    Ok(())
}

fn percentile(samples: &[f64], quantile: f64) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let mut values = samples.to_vec();
    values.sort_unstable_by(f64::total_cmp);
    values
        .get(((values.len() - 1) as f64 * quantile).round() as usize)
        .copied()
}

fn monotonic_us() -> u64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    u64::try_from(START.get_or_init(Instant::now).elapsed().as_micros()).unwrap_or(u64::MAX)
}
