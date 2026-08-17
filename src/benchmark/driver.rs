use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use serde::Serialize;
use tokio::sync::oneshot;

use crate::{
    engine::{Engine, PreparedRequest, ServingHandle, ServingOwnerReport},
    generation::{DEFAULT_SAMPLING_SEED, SamplingConfig},
    scheduler::HardwareCostModel,
};

use super::{
    ArrivalPattern, ArrivalSchedule, RequestObservation, ServingSummary, ServingWorkload,
    standard_workload_matrix,
};

#[derive(Debug, Serialize)]
pub struct ServingScenarioReport {
    pub label: String,
    pub workload: ServingWorkload,
    pub arrival_pattern: &'static str,
    pub observations: usize,
    pub summary: ServingSummary,
}

#[derive(Debug, Serialize)]
pub struct ServingLoadBenchmarkReport {
    pub schema_version: u32,
    pub design: &'static str,
    pub gpu_name: String,
    pub page_size: usize,
    pub scenarios: Vec<ServingScenarioReport>,
    pub owner: ServingOwnerReport,
}

pub fn run_serving_load_benchmark(
    engine: Engine,
    cost_model: HardwareCostModel,
) -> Result<ServingLoadBenchmarkReport> {
    let gpu_name = cost_model.gpu_name.clone();
    let config = engine.continuous_config(cost_model.clone())?;
    let page_size = engine.page_size();
    let (handle, receiver) = ServingHandle::channel(config.queue_capacity);
    let (ready_sender, ready_receiver) = std::sync::mpsc::sync_channel(1);
    let (result_sender, result_receiver) = std::sync::mpsc::sync_channel(1);
    let benchmark_thread = std::thread::Builder::new()
        .name("llm-serving-load-driver".to_string())
        .spawn(move || {
            let result = (|| -> Result<Vec<ServingScenarioReport>> {
                ready_receiver
                    .recv()
                    .context("GPU owner stopped before load benchmark")?;
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_time()
                    .build()
                    .context("failed to start load-driver runtime")?;
                runtime.block_on(run_scenarios(handle, &cost_model))
            })();
            let _ = result_sender.send(result);
        })
        .context("failed to spawn serving load driver")?;
    let owner = engine.run_continuous_owner(config, receiver, ready_sender)?;
    let scenarios = result_receiver
        .recv()
        .context("load driver stopped without a report")??;
    benchmark_thread
        .join()
        .map_err(|_| anyhow::anyhow!("serving load driver panicked"))?;
    Ok(ServingLoadBenchmarkReport {
        schema_version: 1,
        design: "continuous_decode_first_edf_dynamic_chunked_prefill",
        gpu_name,
        page_size,
        scenarios,
        owner,
    })
}

async fn run_scenarios(
    handle: ServingHandle,
    cost_model: &HardwareCostModel,
) -> Result<Vec<ServingScenarioReport>> {
    let mut reports = Vec::new();
    let mut request_id = 1u64;
    for workload in standard_workload_matrix() {
        let request_count = workload.concurrency.max(20);
        let prompts = vec![workload.prompt_tokens; request_count];
        reports.push(
            run_burst_scenario(&handle, workload, &prompts, &mut request_id, cost_model).await?,
        );
    }
    for concurrency in [16usize, 64] {
        let request_count = concurrency.max(20);
        let prompts = (0..request_count)
            .map(|index| if index % 5 == 4 { 2048 } else { 32 })
            .collect::<Vec<_>>();
        reports.push(
            run_burst_scenario(
                &handle,
                ServingWorkload {
                    prompt_tokens: 0,
                    completion_tokens: 128,
                    concurrency,
                },
                &prompts,
                &mut request_id,
                cost_model,
            )
            .await?,
        );
    }
    let poisson_prompts = (0..64)
        .map(|index| if index % 5 == 4 { 2048 } else { 32 })
        .collect::<Vec<_>>();
    reports.push(
        run_poisson_scenario(
            &handle,
            &poisson_prompts,
            128,
            20.0,
            &mut request_id,
            cost_model,
        )
        .await?,
    );
    drop(handle);
    Ok(reports)
}

async fn run_burst_scenario(
    handle: &ServingHandle,
    workload: ServingWorkload,
    prompt_lengths: &[usize],
    request_id: &mut u64,
    cost_model: &HardwareCostModel,
) -> Result<ServingScenarioReport> {
    let started = Instant::now();
    let mut observations = Vec::with_capacity(prompt_lengths.len());
    for wave in prompt_lengths.chunks(workload.concurrency) {
        let mut receivers = Vec::with_capacity(wave.len());
        for &prompt_tokens in wave {
            submit(
                handle,
                prompt_tokens,
                workload.completion_tokens,
                request_id,
                &mut observations,
                &mut receivers,
            );
        }
        collect(receivers, &mut observations).await;
    }
    let wall_seconds = started.elapsed().as_secs_f64();
    let summary = ServingSummary::from_observations(
        &observations,
        cost_model.ttft_slo_ms,
        cost_model.tpot_slo_ms,
        wall_seconds,
    );
    Ok(ServingScenarioReport {
        label: if workload.prompt_tokens == 0 {
            "mixed_80_short_20_long".to_string()
        } else {
            format!(
                "prompt{}_completion{}_concurrency{}",
                workload.prompt_tokens, workload.completion_tokens, workload.concurrency
            )
        },
        workload,
        arrival_pattern: "closed_loop_bursts",
        observations: observations.len(),
        summary,
    })
}

async fn run_poisson_scenario(
    handle: &ServingHandle,
    prompt_lengths: &[usize],
    completion_tokens: usize,
    requests_per_second: f64,
    request_id: &mut u64,
    cost_model: &HardwareCostModel,
) -> Result<ServingScenarioReport> {
    let started = Instant::now();
    let mut observations = Vec::with_capacity(prompt_lengths.len());
    let mut receivers = Vec::with_capacity(prompt_lengths.len());
    let mut arrivals = ArrivalSchedule::new(ArrivalPattern::Poisson {
        requests_per_second,
        seed: DEFAULT_SAMPLING_SEED,
    })?;
    let mut previous_us = 0u64;
    for &prompt_tokens in prompt_lengths {
        let arrival_us = arrivals.next_arrival_us();
        tokio::time::sleep(Duration::from_micros(
            arrival_us.saturating_sub(previous_us),
        ))
        .await;
        previous_us = arrival_us;
        submit(
            handle,
            prompt_tokens,
            completion_tokens,
            request_id,
            &mut observations,
            &mut receivers,
        );
    }
    collect(receivers, &mut observations).await;
    let wall_seconds = started.elapsed().as_secs_f64();
    let summary = ServingSummary::from_observations(
        &observations,
        cost_model.ttft_slo_ms,
        cost_model.tpot_slo_ms,
        wall_seconds,
    );
    Ok(ServingScenarioReport {
        label: "mixed_80_short_20_long_poisson_20rps".to_string(),
        workload: ServingWorkload {
            prompt_tokens: 0,
            completion_tokens,
            concurrency: 0,
        },
        arrival_pattern: "poisson_20rps",
        observations: observations.len(),
        summary,
    })
}

fn submit(
    handle: &ServingHandle,
    prompt_tokens: usize,
    completion_tokens: usize,
    request_id: &mut u64,
    observations: &mut Vec<RequestObservation>,
    receivers: &mut Vec<(
        usize,
        oneshot::Receiver<Result<crate::engine::ServingCompletion, crate::engine::ServingError>>,
    )>,
) {
    let mut token_ids = vec![42u32; prompt_tokens];
    if let Some(first) = token_ids.first_mut() {
        *first = 1;
    }
    let (response, receiver) = oneshot::channel();
    let prepared = PreparedRequest {
        request_id: *request_id,
        token_ids,
        maximum_new_tokens: completion_tokens,
        stop_on_eos: false,
        sampling: SamplingConfig {
            temperature: 0.0,
            top_k: 1,
            repetition_penalty: 1.0,
            seed: *request_id,
        },
        arrived: Instant::now(),
        response,
    };
    *request_id = request_id.saturating_add(1);
    if handle.try_submit(prepared).is_ok() {
        receivers.push((prompt_tokens, receiver));
    } else {
        observations.push(RequestObservation {
            ttft_ms: 0.0,
            tpot_ms: 0.0,
            queue_delay_ms: 0.0,
            prompt_tokens,
            output_tokens: 0,
            accepted: false,
        });
    }
}

async fn collect(
    receivers: Vec<(
        usize,
        oneshot::Receiver<Result<crate::engine::ServingCompletion, crate::engine::ServingError>>,
    )>,
    observations: &mut Vec<RequestObservation>,
) {
    for (prompt_tokens, receiver) in receivers {
        match receiver.await {
            Ok(Ok(completion)) => observations.push(RequestObservation {
                ttft_ms: completion.metrics.ttft_ms,
                tpot_ms: completion.metrics.tpot_p95_ms.unwrap_or(0.0),
                queue_delay_ms: completion.metrics.queue_delay_ms,
                prompt_tokens,
                output_tokens: completion.token_ids.len(),
                accepted: true,
            }),
            Ok(Err(_)) | Err(_) => observations.push(RequestObservation {
                ttft_ms: 0.0,
                tpot_ms: 0.0,
                queue_delay_ms: 0.0,
                prompt_tokens,
                output_tokens: 0,
                accepted: false,
            }),
        }
    }
}
