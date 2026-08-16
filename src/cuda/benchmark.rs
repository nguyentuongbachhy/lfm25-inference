use std::sync::Arc;

use anyhow::Result;
use cudarc::driver::{CudaContext, CudaStream, sys};

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
        start.record(stream)?;

        for _ in 0..config.iterations_per_batch {
            run()?;
        }

        end.record(stream)?;
        end.synchronize()?;

        let elapsed_ms = start.elapsed_ms(&end)?;

        let per_iteration_us = f64::from(elapsed_ms) * 1000.0 / config.iterations_per_batch as f64;

        samples.push(per_iteration_us);
    }

    samples.sort_by(f64::total_cmp);

    let mean = samples.iter().sum::<f64>() / samples.len() as f64;

    Ok(BenchStats {
        mean_us: mean,
        p50_us: percentile(&samples, 0.50),
        p95_us: percentile(&samples, 0.95),
        min_us: samples[0],
    })
}

fn percentile(samples: &[f64], percentile: f64) -> f64 {
    let index = ((samples.len() - 1) as f64 * percentile).round() as usize;

    samples[index]
}
