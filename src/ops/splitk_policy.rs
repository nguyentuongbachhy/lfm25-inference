const SPLITK_MAX_SPLITS: usize = 8;
const LEGACY_TARGET_BLOCKS: usize = 64;
const LEGACY_MIN_PAGES_PER_SPLIT: usize = 8;
const LEGACY_MIN_CONTEXT_TOKENS: usize = 1024;
const PS16_MIN_MEASURED_CONTEXT_TOKENS: usize = 512;

/// Select the number of KV-axis splits for decode attention.
///
/// Page-size 16 is tuned from paired GPU measurements on the target RTX 5060
/// Laptop GPU. The policy is deliberately conservative where the measured gain
/// was small or noisy, and returns one to select the existing unsplit kernel.
///
/// Page-size 32 retains the previous analytic policy until it has an equivalent
/// hardware sweep; PS16 measurements are not assumed to transfer across layouts.
pub(crate) fn splitk_decode_splits(
    num_tokens: usize,
    maximum_context_tokens: usize,
    page_size: usize,
) -> usize {
    if num_tokens == 0 {
        return 1;
    }

    match page_size {
        16 => splitk_decode_splits_ps16(num_tokens, maximum_context_tokens),
        32 => splitk_decode_splits_ps32_legacy(num_tokens, maximum_context_tokens),
        _ => 1,
    }
}

#[inline]
fn splitk_decode_splits_ps16(num_tokens: usize, context_tokens: usize) -> usize {
    if context_tokens < PS16_MIN_MEASURED_CONTEXT_TOKENS {
        return 1;
    }

    match num_tokens {
        // B1: four splits is materially more stable at C512; from C1024 the
        // eight-way kernel wins decisively.
        1 => {
            if context_tokens < 1024 {
                4
            } else {
                8
            }
        }
        // B2 has a measured occupancy crossover: 8-way at C512, 4-way through
        // C2048, then 8-way again once KV work dominates at C4096+.
        2 => {
            if context_tokens < 1024 {
                8
            } else if context_tokens < 4096 {
                4
            } else {
                8
            }
        }
        // B3-B8 consistently prefer eight splits across C512-C8192.
        3..=8 => 8,
        // B16 prefers four splits through C2048. At C4096 eight splits only
        // improves mean by <1% and has no p95 advantage, so keep four.
        9..=16 => 4,
        // B32 has enough native occupancy that splitting is only worthwhile
        // once context becomes material. The C512 two-way gain was ~2%, below
        // the production threshold, so retain the unsplit path there.
        17..=32 => {
            if context_tokens < 1024 {
                1
            } else if context_tokens < 2048 {
                2
            } else {
                4
            }
        }
        // B64 measurements were neutral-to-negative for all split counts.
        _ => 1,
    }
}

#[inline]
fn splitk_decode_splits_ps32_legacy(num_tokens: usize, context_tokens: usize) -> usize {
    if context_tokens < LEGACY_MIN_CONTEXT_TOKENS {
        return 1;
    }
    let base_blocks = num_tokens.saturating_mul(8).max(1);
    let occupancy_splits = LEGACY_TARGET_BLOCKS
        .div_ceil(base_blocks)
        .clamp(1, SPLITK_MAX_SPLITS);
    let context_pages = context_tokens.div_ceil(32);
    let page_splits = (context_pages / LEGACY_MIN_PAGES_PER_SPLIT).clamp(1, SPLITK_MAX_SPLITS);
    occupancy_splits.min(page_splits).max(1)
}

#[cfg(test)]
mod tests {
    use super::splitk_decode_splits;

    #[test]
    fn ps16_dispatch_matches_measured_buckets() {
        assert_eq!(splitk_decode_splits(1, 511, 16), 1);
        assert_eq!(splitk_decode_splits(1, 512, 16), 4);
        assert_eq!(splitk_decode_splits(1, 1024, 16), 8);

        assert_eq!(splitk_decode_splits(2, 512, 16), 8);
        assert_eq!(splitk_decode_splits(2, 1024, 16), 4);
        assert_eq!(splitk_decode_splits(2, 2048, 16), 4);
        assert_eq!(splitk_decode_splits(2, 4096, 16), 8);

        assert_eq!(splitk_decode_splits(4, 512, 16), 8);
        assert_eq!(splitk_decode_splits(8, 2048, 16), 8);
        assert_eq!(splitk_decode_splits(16, 512, 16), 4);
        assert_eq!(splitk_decode_splits(16, 4096, 16), 4);

        assert_eq!(splitk_decode_splits(32, 512, 16), 1);
        assert_eq!(splitk_decode_splits(32, 1024, 16), 2);
        assert_eq!(splitk_decode_splits(32, 2048, 16), 4);
        assert_eq!(splitk_decode_splits(33, 4096, 16), 1);
        assert_eq!(splitk_decode_splits(64, 2048, 16), 1);
    }

    #[test]
    fn ps32_retains_legacy_policy_until_measured() {
        assert_eq!(splitk_decode_splits(1, 512, 32), 1);
        assert_eq!(splitk_decode_splits(1, 1024, 32), 4);
        assert_eq!(splitk_decode_splits(1, 2048, 32), 8);
        assert_eq!(splitk_decode_splits(2, 2048, 32), 4);
        assert_eq!(splitk_decode_splits(4, 2048, 32), 2);
        assert_eq!(splitk_decode_splits(8, 2048, 32), 1);
    }

    #[test]
    fn unsupported_or_empty_inputs_stay_unsplit() {
        assert_eq!(splitk_decode_splits(0, 2048, 16), 1);
        assert_eq!(splitk_decode_splits(1, 2048, 8), 1);
    }
}
