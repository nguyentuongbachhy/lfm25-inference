use std::{
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result, ensure};
use half::bf16;
use tokio::sync::mpsc;

use crate::{
    cache::PageRadixCache,
    model::{ConvCheckpointPool, RaggedBatchInput},
    ops,
    scheduler::{
        RequestInit, RequestPhase, RequestSlotId, RequestSlots, ScheduledWork, Scheduler,
    },
};

use super::{
    ContinuousEngineConfig, Engine, PreparedRequest, ResponseState, ServingError, ServingOwnerReport,
    finish_request, monotonic_us, warm_serving_path,
};

const RADIX_KV_BUDGET_DIVISOR: usize = 4;
const CONV_CHECKPOINT_CAPACITY: usize = 64;

pub(super) fn run_owner_radix(
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

    let prefix_page_budget = (config.physical_kv_pages / RADIX_KV_BUDGET_DIVISOR).max(1);
    let mut radix = PageRadixCache::new(engine.config.kv_page_size, prefix_page_budget)?;
    let convolution_layers = engine
        .model
        .config()
        .layer_types
        .iter()
        .filter(|kind| kind.as_str() != "full_attention")
        .count();
    let mut checkpoints = ConvCheckpointPool::new(
        &engine.runtime,
        convolution_layers,
        CONV_CHECKPOINT_CAPACITY,
        engine.model.config().hidden_size,
        engine.model.config().conv_l_cache - 1,
    )?;
    let mut matched_pages = Vec::with_capacity(
        config
            .maximum_sequence_tokens
            .div_ceil(engine.config.kv_page_size.value()),
    );

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
            admit_request_radix(
                &engine,
                &config,
                &mut cache,
                &mut slots,
                &mut responses,
                &mut radix,
                &checkpoints,
                &mut matched_pages,
                request,
            )?;
        }
        if slots.free_count() == config.maximum_request_slots {
            let Some(request) = receiver.blocking_recv() else {
                break;
            };
            admit_request_radix(
                &engine,
                &config,
                &mut cache,
                &mut slots,
                &mut responses,
                &mut radix,
                &checkpoints,
                &mut matched_pages,
                request,
            )?;
            continue;
        }

        refresh_unscheduled_prefixes(
            &engine,
            &mut cache,
            &mut slots,
            &responses,
            &mut radix,
            &checkpoints,
            &mut matched_pages,
        )?;

        maximum_active_requests = maximum_active_requests.max(
            config
                .maximum_request_slots
                .saturating_sub(slots.free_count()),
        );

        let scheduler_started = Instant::now();
        let now_us = monotonic_us();
        work.clear();
        work.extend_from_slice(scheduler.schedule(&slots, now_us).work());
        split_prefill_at_reusable_boundary(&mut work, &slots, engine.config.kv_page_size.value())?;
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

        enqueue_sampled_download(&engine, &sampled, &mut sampled_host)?;
        let copy_finished = engine.runtime.record_timing_event()?;
        // Synchronizing the copy-complete event waits for both model execution
        // and the following D2H copy. The GPU-complete event is therefore ready
        // when its interval is queried below, removing the host scheduling gap
        // between the two operations without changing their measured boundaries.
        let d2h_ms = engine.runtime.elapsed_ms(&gpu_finished, &copy_finished)?;
        let gpu_ms = engine.runtime.elapsed_ms(&gpu_started, &gpu_finished)?;
        let sampled = sampled_host
            .as_slice()
            .context("failed to synchronize pinned token output")?;
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
        let token_ready = Instant::now();
        let step_ms = token_ready.duration_since(step_started).as_secs_f64() * 1000.0;
        if config.trace_steps
            && work
                .iter()
                .any(|item| matches!(item, ScheduledWork::Prefill { .. }))
        {
            eprintln!(
                "radix_serving_step tokens={} segments={} wall_ms={step_ms:.3} submit_cpu_ms={submit_cpu_ms:.3} gpu_ms={gpu_ms:.3} d2h_ms={d2h_ms:.3} bf16_hits={bf16_hits} bf16_misses={bf16_misses} bf16_retained={} bf16_dropped={}",
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

                if request.prefilled > 0
                    && request
                        .prefilled
                        .is_multiple_of(engine.config.kv_page_size.value())
                {
                    cache.publish_prefix_checkpoint(
                        &engine.runtime,
                        slot.0 as usize,
                        request.tokens(),
                        request.prefilled,
                        &mut radix,
                        &mut checkpoints,
                    )?;
                }
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
                    .push(token_ready.duration_since(last).as_secs_f64() * 1000.0);
            } else {
                state.first_token_ready = Some(token_ready);
            }
            state.last_token_ready = Some(token_ready);
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

    let prefix = radix.snapshot();
    eprintln!(
        "radix cache: hits={} misses={} matched_tokens={} nodes={} cached_pages={} checkpoints={} checkpoint_slots_used={}/{}",
        prefix.hits,
        prefix.misses,
        prefix.matched_tokens,
        prefix.nodes,
        prefix.cached_pages,
        prefix.checkpoints,
        checkpoints.capacity().saturating_sub(checkpoints.available()),
        checkpoints.capacity(),
    );

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

fn enqueue_sampled_download(
    engine: &Engine,
    sampled: &crate::tensor::Tensor<u32>,
    destination: &mut cudarc::driver::PinnedHostSlice<u32>,
) -> Result<()> {
    ensure!(
        sampled.numel() <= destination.len(),
        "pinned output ring is too small"
    );
    let logical = sampled
        .storage()
        .try_slice(0..sampled.numel())
        .context("invalid logical u32 download range")?;
    let destination = destination
        .as_mut_slice()
        .context("failed to access pinned token output")?;
    engine
        .runtime
        .stream()
        .memcpy_dtoh(&logical, &mut destination[..sampled.numel()])
        .context("failed to enqueue sampled token download")
}

#[allow(clippy::too_many_arguments)]
fn refresh_unscheduled_prefixes(
    engine: &Engine,
    cache: &mut crate::model::BatchModelCache,
    slots: &mut RequestSlots,
    responses: &[ResponseState],
    radix: &mut PageRadixCache,
    checkpoints: &ConvCheckpointPool,
    matched_pages: &mut Vec<u32>,
) -> Result<()> {
    let page_size = engine.config.kv_page_size.value();
    let slot_count = slots.entries().len();
    for index in 0..slot_count {
        if responses[index].first_scheduled.is_some() {
            continue;
        }
        let slot = RequestSlotId(u32::try_from(index)?);
        let (phase, current_prefix, reusable_limit, hit) = {
            let request = slots.get(slot)?;
            if request.phase != RequestPhase::QueuedPrefill {
                continue;
            }
            let reusable_limit = request
                .prompt_len
                .saturating_sub(1)
                .checked_div(page_size)
                .unwrap_or(0)
                .saturating_mul(page_size);
            if reusable_limit <= request.prefilled {
                continue;
            }
            let hit = radix.probe_checkpoint(request.tokens(), reusable_limit, matched_pages);
            (request.phase, request.prefilled, reusable_limit, hit)
        };
        if phase != RequestPhase::QueuedPrefill || reusable_limit <= current_prefix {
            continue;
        }
        let Some(hit) = hit else {
            continue;
        };
        if hit.token_len <= current_prefix {
            continue;
        }

        let released = cache.extend_attached_prefix(
            &engine.runtime,
            index,
            current_prefix,
            hit.token_len,
            matched_pages,
            checkpoints,
            hit.checkpoint_slot,
        )?;
        let request = slots.get_mut(slot)?;
        request.prefilled = hit.token_len;
        request.reserved_pages = request.reserved_pages.saturating_sub(released);
    }
    Ok(())
}

fn split_prefill_at_reusable_boundary(
    work: &mut [ScheduledWork],
    slots: &RequestSlots,
    page_size: usize,
) -> Result<()> {
    for item in work {
        let ScheduledWork::Prefill { slot, tokens } = item else {
            continue;
        };
        let request = slots.get(*slot)?;
        let reusable_limit = request
            .prompt_len
            .saturating_sub(1)
            .checked_div(page_size)
            .unwrap_or(0)
            .saturating_mul(page_size);
        if request.prefilled >= reusable_limit || reusable_limit == 0 {
            continue;
        }
        let scheduled_end = request.prefilled.saturating_add(*tokens);
        if scheduled_end > reusable_limit {
            *tokens = reusable_limit - request.prefilled;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn admit_request_radix(
    engine: &Engine,
    config: &ContinuousEngineConfig,
    cache: &mut crate::model::BatchModelCache,
    slots: &mut RequestSlots,
    responses: &mut [ResponseState],
    radix: &mut PageRadixCache,
    checkpoints: &ConvCheckpointPool,
    matched_pages: &mut Vec<u32>,
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

    let page_size = engine.config.kv_page_size.value();
    let reusable_limit = request
        .token_ids
        .len()
        .saturating_sub(1)
        .checked_div(page_size)
        .unwrap_or(0)
        .saturating_mul(page_size);
    let prefix_hit = radix.longest_checkpoint(
        &request.token_ids,
        reusable_limit,
        matched_pages,
    );
    let prefix_tokens = prefix_hit.map_or(0, |hit| hit.token_len);
    let remaining_prompt = request.token_ids.len().saturating_sub(prefix_tokens);

    let mut decode_count = 1usize;
    let mut maximum_context = total;
    let mut predicted_prefill_queue_ms = config.cost_model.predict_prefill_ms(remaining_prompt);
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

    let total_pages = total.div_ceil(page_size);
    let shared_pages = prefix_tokens / page_size;
    let private_pages = total_pages.saturating_sub(shared_pages);
    if let Err(error) = cache.reserve_private_pages(slot.0 as usize, private_pages) {
        slots.release(slot)?;
        let _ = request.response.send(Err(ServingError::overloaded(format!(
            "KV admission rejected: {error}"
        ))));
        return Ok(());
    }

    slots.get_mut(slot)?.initialize(RequestInit::new(
        request.request_id,
        &request.token_ids,
        total,
        monotonic_us(),
        config.scheduler.ttft_slo_us,
        config.scheduler.tpot_slo_us,
        private_pages,
    ))?;

    if let Some(hit) = prefix_hit {
        if let Err(error) = cache.attach_prefix(
            &engine.runtime,
            slot.0 as usize,
            hit.token_len,
            matched_pages,
            checkpoints,
            hit.checkpoint_slot,
        ) {
            cache.release(&engine.runtime, slot.0 as usize)?;
            slots.release(slot)?;
            let _ = request.response.send(Err(ServingError::overloaded(format!(
                "prefix cache attach failed: {error}"
            ))));
            return Ok(());
        }
        slots.get_mut(slot)?.prefilled = hit.token_len;
    }

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
