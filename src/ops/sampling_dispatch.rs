use std::sync::OnceLock;

#[cfg(test)]
use std::sync::atomic::{AtomicI8, Ordering};

use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cuda::CudaRuntime,
    tensor::{Shape, Tensor},
};

use super::sampling;

// Full-model ABBA on the target RTX 5060 showed stable mean/p95 wins through
// B=16. B=32 was context-dependent and B=64 regressed materially, so the
// production dispatch boundary is intentionally conservative even though the
// packed atomic implementation is functionally valid at larger row counts.
const ATOMIC_ARGMAX_MAX_ROWS: usize = 16;
const ATOMIC_ARGMAX_MAX_COLUMNS: usize = 65_536;

static ATOMIC_ARGMAX_ENABLED: OnceLock<bool> = OnceLock::new();

#[cfg(test)]
static ATOMIC_ARGMAX_TEST_OVERRIDE: AtomicI8 = AtomicI8::new(-1);

fn atomic_argmax_enabled_from_env() -> bool {
    *ATOMIC_ARGMAX_ENABLED.get_or_init(|| {
        std::env::var("LFM25_ATOMIC_ARGMAX")
            .map(|value| {
                !matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
            })
            .unwrap_or(true)
    })
}

#[inline]
fn atomic_argmax_enabled() -> bool {
    #[cfg(test)]
    {
        match ATOMIC_ARGMAX_TEST_OVERRIDE.load(Ordering::Relaxed) {
            0 => return false,
            1 => return true,
            _ => {}
        }
    }
    atomic_argmax_enabled_from_env()
}

#[inline]
fn should_use_atomic_argmax(rows: usize, columns: usize, enabled: bool) -> bool {
    enabled
        && rows > 0
        && rows <= ATOMIC_ARGMAX_MAX_ROWS
        && columns > 0
        && columns <= ATOMIC_ARGMAX_MAX_COLUMNS
}

pub fn argmax_rows_bf16(runtime: &CudaRuntime, input: &Tensor<bf16>) -> Result<Tensor<u32>> {
    ensure!(input.rank() == 2, "batched argmax expects rank-2 input");
    let rows = input.dims()[0];
    let mut output = runtime.alloc_u32(Shape::new([rows]))?;
    argmax_rows_bf16_into(runtime, input, &mut output)?;
    Ok(output)
}

pub fn argmax_bf16(runtime: &CudaRuntime, input: &Tensor<bf16>) -> Result<Tensor<u32>> {
    ensure!(input.numel() > 0, "argmax does not support empty tensors");
    let numel = input.numel();
    let mut output = runtime.alloc_uninit::<u32>(Shape::new([1]))?;
    if should_use_atomic_argmax(1, numel, atomic_argmax_enabled()) {
        output.set_logical_shape(Shape::new([1]))?;
        unsafe {
            runtime
                .kernels()
                .sampling()
                .launch_argmax_rows_bf16_atomic(
                    runtime.stream(),
                    input.storage(),
                    output.storage_mut(),
                    1,
                    numel,
                )?;
        }
    } else {
        unsafe {
            runtime.kernels().sampling().launch_argmax_bf16(
                runtime.stream(),
                input.storage(),
                output.storage_mut(),
                numel,
            )?;
        }
    }
    Ok(output)
}

pub(crate) fn argmax_rows_bf16_into(
    runtime: &CudaRuntime,
    input: &Tensor<bf16>,
    output: &mut Tensor<u32>,
) -> Result<()> {
    let use_atomic = input.rank() == 2
        && should_use_atomic_argmax(input.dims()[0], input.dims()[1], atomic_argmax_enabled());

    if use_atomic {
        sampling::argmax_rows_bf16_atomic_into(runtime, input, output)
    } else {
        sampling::argmax_rows_bf16_into(runtime, input, output)
    }
}

#[cfg(test)]
pub(crate) fn set_atomic_argmax_test_override(enabled: Option<bool>) {
    let value = match enabled {
        Some(false) => 0,
        Some(true) => 1,
        None => -1,
    };
    ATOMIC_ARGMAX_TEST_OVERRIDE.store(value, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::benchmark::{BenchConfig, benchmark_gpu_paired};

    #[test]
    fn production_policy_is_atomic_only_through_batch_16() {
        assert!(should_use_atomic_argmax(1, 65_536, true));
        assert!(should_use_atomic_argmax(16, 65_536, true));
        assert!(!should_use_atomic_argmax(17, 65_536, true));
        assert!(!should_use_atomic_argmax(32, 65_536, true));
        assert!(!should_use_atomic_argmax(64, 65_536, true));
    }

    #[test]
    fn production_policy_falls_back_outside_packing_domain_or_when_disabled() {
        assert!(!should_use_atomic_argmax(0, 65_536, true));
        assert!(!should_use_atomic_argmax(1, 0, true));
        assert!(!should_use_atomic_argmax(1, 65_537, true));
        assert!(!should_use_atomic_argmax(1, 65_536, false));
    }

    #[test]
    fn argmax_dispatch_matches_reference() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let columns = 32_768usize;
        for rows in [1, 4, 16] {
            let mut host_logits = Vec::with_capacity(rows * columns);
            for r in 0..rows {
                for c in 0..columns {
                    let val = ((r * 1000 + c) as f32 * 0.001).sin();
                    host_logits.push(bf16::from_f32(val));
                }
            }
            let input = runtime.upload(&host_logits, Shape::new([rows, columns]))?;

            // Legacy
            set_atomic_argmax_test_override(Some(false));
            let legacy_out = argmax_rows_bf16(&runtime, &input)?;
            let legacy_idx = runtime.download(&legacy_out)?;

            // Atomic
            set_atomic_argmax_test_override(Some(true));
            let atomic_out = argmax_rows_bf16(&runtime, &input)?;
            let atomic_idx = runtime.download(&atomic_out)?;

            set_atomic_argmax_test_override(None);
            assert_eq!(legacy_idx, atomic_idx, "mismatch at rows={rows}");

            if rows == 1 {
                set_atomic_argmax_test_override(Some(true));
                let single_out = argmax_bf16(&runtime, &input)?;
                let single_idx = runtime.download(&single_out)?;
                set_atomic_argmax_test_override(None);
                assert_eq!(legacy_idx, single_idx, "single argmax mismatch");
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "GPU benchmark: paired ABBA argmax dispatch"]
    fn bench_argmax_dispatch_paired_abba() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let columns = 65_536usize;

        for rows in [1, 4, 16] {
            let mut host_logits = Vec::with_capacity(rows * columns);
            for r in 0..rows {
                for c in 0..columns {
                    let val = ((r * 1337 + c) as f32 * 0.0013).sin();
                    host_logits.push(bf16::from_f32(val));
                }
            }
            let input = runtime.upload(&host_logits, Shape::new([rows, columns]))?;
            let mut ref_out = runtime.alloc_u32(Shape::new([rows]))?;
            let mut cand_out = runtime.alloc_u32(Shape::new([rows]))?;

            let stats = benchmark_gpu_paired(
                runtime.context(),
                runtime.stream(),
                BenchConfig {
                    warmup: 2,
                    batches: 10,
                    iterations_per_batch: 5,
                },
                || {
                    set_atomic_argmax_test_override(Some(false));
                    argmax_rows_bf16_into(&runtime, &input, &mut ref_out)?;
                    Ok(())
                },
                || {
                    set_atomic_argmax_test_override(Some(true));
                    argmax_rows_bf16_into(&runtime, &input, &mut cand_out)?;
                    Ok(())
                },
            )?;

            set_atomic_argmax_test_override(None);

            let ref_download = runtime.download(&ref_out)?;
            let cand_download = runtime.download(&cand_out)?;
            assert_eq!(ref_download, cand_download);

            println!(
                "argmax_dispatch_abba rows={} cols={} legacy_us={:.2} atomic_us={:.2} speedup_mean={:.4}x speedup_p50={:.4}x speedup_p95={:.4}x",
                rows,
                columns,
                stats.reference.mean_us,
                stats.candidate.mean_us,
                stats.speedup_mean,
                stats.speedup_p50,
                stats.speedup_p95,
            );

            ensure!(
                stats.speedup_mean >= 1.50,
                "rows={rows} failed primitive speedup gate: {:.4}x < 1.50x",
                stats.speedup_mean
            );
        }

        Ok(())
    }
}
