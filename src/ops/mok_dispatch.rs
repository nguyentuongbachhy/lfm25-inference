/// Select the one-kernel MoK decode path for the measured short-context region.
///
/// The policy is intentionally monotonic and conservative. It comes from the
/// paired ragged decode benchmark on the target RTX 5060 Laptop GPU. Outside
/// these regions the two-kernel path (fused Q/K postprocess + W8 fast-exp
/// paged attention) is the production default.
pub(crate) fn should_use_mok_one_kernel(
    page_size: usize,
    context_tokens: usize,
    batch_size: usize,
) -> bool {
    if context_tokens == 0 || batch_size == 0 {
        return false;
    }

    match page_size {
        16 => {
            context_tokens <= 16
                || (context_tokens <= 32 && batch_size <= 32)
                || (context_tokens <= 64 && batch_size <= 8)
        }
        32 => {
            (context_tokens <= 32 && batch_size <= 16) || (context_tokens <= 64 && batch_size <= 8)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::should_use_mok_one_kernel;

    #[test]
    fn ps16_dispatch_boundaries_match_measured_policy() {
        assert!(should_use_mok_one_kernel(16, 16, 64));
        assert!(should_use_mok_one_kernel(16, 32, 32));
        assert!(!should_use_mok_one_kernel(16, 32, 64));
        assert!(should_use_mok_one_kernel(16, 64, 8));
        assert!(!should_use_mok_one_kernel(16, 64, 16));
        assert!(!should_use_mok_one_kernel(16, 128, 1));
    }

    #[test]
    fn ps32_dispatch_boundaries_match_measured_policy() {
        assert!(should_use_mok_one_kernel(32, 16, 16));
        assert!(!should_use_mok_one_kernel(32, 16, 32));
        assert!(should_use_mok_one_kernel(32, 32, 16));
        assert!(!should_use_mok_one_kernel(32, 32, 32));
        assert!(should_use_mok_one_kernel(32, 64, 8));
        assert!(!should_use_mok_one_kernel(32, 64, 16));
        assert!(!should_use_mok_one_kernel(32, 128, 1));
    }

    #[test]
    fn unsupported_or_empty_inputs_use_two_kernel_path() {
        assert!(!should_use_mok_one_kernel(8, 16, 1));
        assert!(!should_use_mok_one_kernel(16, 0, 1));
        assert!(!should_use_mok_one_kernel(16, 16, 0));
    }
}
