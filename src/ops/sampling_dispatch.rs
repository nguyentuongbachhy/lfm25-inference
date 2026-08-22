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

pub(crate) fn argmax_rows_bf16_into(
    runtime: &CudaRuntime,
    input: &Tensor<bf16>,
    output: &mut Tensor<u32>,
) -> Result<()> {
    let use_atomic = input.rank() == 2
        && should_use_atomic_argmax(
            input.dims()[0],
            input.dims()[1],
            atomic_argmax_enabled(),
        );

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
}
