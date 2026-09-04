use std::sync::OnceLock;

#[cfg(test)]
use std::sync::atomic::{AtomicI8, Ordering};

static FLASH_PREFILL_ENABLED: OnceLock<bool> = OnceLock::new();

#[cfg(test)]
static FLASH_PREFILL_TEST_OVERRIDE: AtomicI8 = AtomicI8::new(-1);

fn flash_prefill_enabled_from_env() -> bool {
    *FLASH_PREFILL_ENABLED.get_or_init(|| {
        std::env::var("LFM25_FLASH_PREFILL")
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
pub(crate) fn flash_prefill_enabled() -> bool {
    #[cfg(test)]
    {
        match FLASH_PREFILL_TEST_OVERRIDE.load(Ordering::Relaxed) {
            0 => return false,
            1 => return true,
            _ => {}
        }
    }
    flash_prefill_enabled_from_env()
}

#[inline]
pub(crate) fn should_use_flash_prefill(num_tokens: usize) -> bool {
    flash_prefill_enabled() && num_tokens > 0
}

#[cfg(test)]
pub(crate) struct ScopedFlashPrefillOverride(i8);

#[cfg(test)]
impl ScopedFlashPrefillOverride {
    pub(crate) fn new(enabled: bool) -> Self {
        let previous =
            FLASH_PREFILL_TEST_OVERRIDE.swap(if enabled { 1 } else { 0 }, Ordering::Relaxed);
        Self(previous)
    }
}

#[cfg(test)]
impl Drop for ScopedFlashPrefillOverride {
    fn drop(&mut self) {
        FLASH_PREFILL_TEST_OVERRIDE.store(self.0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_respects_test_override_and_token_count() {
        let _guard = ScopedFlashPrefillOverride::new(true);
        assert!(should_use_flash_prefill(1));
        assert!(should_use_flash_prefill(16));
        assert!(should_use_flash_prefill(2048));
        assert!(!should_use_flash_prefill(0));

        let _guard_off = ScopedFlashPrefillOverride::new(false);
        assert!(!should_use_flash_prefill(1));
        assert!(!should_use_flash_prefill(2048));
    }
}
