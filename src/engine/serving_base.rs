use std::time::Instant;

use anyhow::{Context as _, Result, ensure};
use half::bf16;
use serde::Serialize;
use tokio::sync::{mpsc, oneshot};

use crate::{
    generation::SamplingConfig,
    model::RaggedBatchInput,
    ops,
    scheduler::{HardwareCostModel, RequestSlotId, RequestSlots, SchedulerConfig},
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
        speculative_draft_tokens: 0,
        speculative_accepted_tokens: 0,
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
