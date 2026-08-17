use std::sync::Arc;

use anyhow::Result;
use cudarc::driver::{CudaContext, CudaEvent, CudaStream, sys};

#[derive(Clone, Copy)]
pub(crate) struct BenchConfig {
    pub warmup: usize,
    pub batches: usize,
    pub iterations_per_batch: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            warmup: 100,
            batches: 100,
            iterations_per_batch: 100,
        }
    }
}

#[derive(Debug)]
pub(crate) struct BenchStats {
    pub mean_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub min_us: f64,
}

impl BenchStats {
    pub(crate) fn effective_gbps(&self, bytes_per_iteration: usize) -> f64 {
        bytes_per_iteration as f64 / self.mean_us / 1_000.0
    }

    pub(crate) fn effective_tflops(&self, floating_point_operations: usize) -> f64 {
        floating_point_operations as f64 / self.mean_us / 1_000_000.0
    }
}

#[derive(Debug)]
pub(crate) struct PairedBenchStats {
    pub reference: BenchStats,
    pub candidate: BenchStats,
    pub speedup_mean: f64,
    pub speedup_p50: f64,
    pub speedup_p95: f64,
    pub speedup_min: f64,
    pub speedup_max: f64,
}

fn stats(mut samples: Vec<f64>) -> BenchStats {
    samples.sort_by(f64::total_cmp);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    BenchStats {
        mean_us: mean,
        p50_us: percentile(&samples, 0.50),
        p95_us: percentile(&samples, 0.95),
        min_us: samples[0],
    }
}

fn measure_batch<F>(
    start: &CudaEvent,
    end: &CudaEvent,
    stream: &CudaStream,
    iterations: usize,
    run: &mut F,
) -> Result<f64>
where
    F: FnMut() -> Result<()>,
{
    start.record(stream)?;
    for _ in 0..iterations {
        run()?;
    }
    end.record(stream)?;
    end.synchronize()?;
    Ok(f64::from(start.elapsed_ms(end)?) * 1000.0 / iterations as f64)
}

pub(crate) fn benchmark_gpu<F>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    config: BenchConfig,
    mut run: F,
) -> Result<BenchStats>
where
    F: FnMut() -> Result<()>,
{
    for _ in 0..config.warmup {
        run()?;
    }
    stream.synchronize()?;

    let start = context.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let end = context.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let mut samples = Vec::with_capacity(config.batches);

    for _ in 0..config.batches {
        samples.push(measure_batch(
            &start,
            &end,
            stream,
            config.iterations_per_batch,
            &mut run,
        )?);
    }
    Ok(stats(samples))
}

/// Benchmarks two GPU paths in balanced AB/BA order inside one process.
///
/// Each pair uses the same event objects, stream, iteration count and thermal
/// state. Alternating the execution order reduces laptop boost-clock and power
/// bias that would otherwise favor whichever implementation always runs second.
pub(crate) fn benchmark_gpu_paired<R, C>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    config: BenchConfig,
    mut reference: R,
    mut candidate: C,
) -> Result<PairedBenchStats>
where
    R: FnMut() -> Result<()>,
    C: FnMut() -> Result<()>,
{
    for pair in 0..config.warmup {
        if pair & 1 == 0 {
            reference()?;
            candidate()?;
        } else {
            candidate()?;
            reference()?;
        }
    }
    stream.synchronize()?;

    let start = context.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let end = context.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let mut reference_samples = Vec::with_capacity(config.batches);
    let mut candidate_samples = Vec::with_capacity(config.batches);
    let mut speedups = Vec::with_capacity(config.batches);

    for pair in 0..config.batches {
        let (reference_us, candidate_us) = if pair & 1 == 0 {
            let reference_us = measure_batch(
                &start,
                &end,
                stream,
                config.iterations_per_batch,
                &mut reference,
            )?;
            let candidate_us = measure_batch(
                &start,
                &end,
                stream,
                config.iterations_per_batch,
                &mut candidate,
            )?;
            (reference_us, candidate_us)
        } else {
            let candidate_us = measure_batch(
                &start,
                &end,
                stream,
                config.iterations_per_batch,
                &mut candidate,
            )?;
            let reference_us = measure_batch(
                &start,
                &end,
                stream,
                config.iterations_per_batch,
                &mut reference,
            )?;
            (reference_us, candidate_us)
        };
        reference_samples.push(reference_us);
        candidate_samples.push(candidate_us);
        speedups.push(reference_us / candidate_us);
    }

    let reference = stats(reference_samples);
    let candidate = stats(candidate_samples);
    speedups.sort_by(f64::total_cmp);
    let speedup_mean = speedups.iter().sum::<f64>() / speedups.len() as f64;

    Ok(PairedBenchStats {
        reference,
        candidate,
        speedup_mean,
        speedup_p50: percentile(&speedups, 0.50),
        speedup_p95: percentile(&speedups, 0.95),
        speedup_min: speedups[0],
        speedup_max: speedups[speedups.len() - 1],
    })
}

fn percentile(samples: &[f64], percentile: f64) -> f64 {
    let index = ((samples.len() - 1) as f64 * percentile).round() as usize;
    samples[index]
}
