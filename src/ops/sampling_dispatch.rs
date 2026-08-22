use std::sync::OnceLock;

#[cfg(test)]
use std::sync::atomic::{AtomicI8, Ordering};

use anyhow::Result;
use half::bf16;

use crate::{
    cuda::CudaRuntime,
    tensor::Tensor,
};

use super::sampling;

// Full-model ABBA on the target RTX 5060 showed stable mean/p95 wins through
// B=16. B=32 was context-dependent and B=64 regressed materially, so keep the
// measured production boundary conservative even though the atomic kernel is
// functionally valid up to 64 rows.
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

pub(crate) fn argmax_rows_bf16_into(
    runtime: &CudaRuntime,
    input: &Tensor<bf16>,
    output: &mut Tensor<u32>,
) -> Result<()> {
    let use_atomic = atomic_argmax_enabled()
        && input.rank() == 2
        && input.dims()[0] <= ATOMIC_ARGMAX_MAX_ROWS
        && input.dims()[1] <= ATOMIC_ARGMAX_MAX_COLUMNS;

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
