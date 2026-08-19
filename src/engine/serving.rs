include!("serving_base.rs");

mod radix_owner;

impl Engine {
    /// Continuous serving with page-granular radix prefix reuse and matching
    /// GPU-resident recurrent convolution checkpoints.
    pub fn run_continuous_owner_radix(
        self,
        config: ContinuousEngineConfig,
        receiver: mpsc::Receiver<PreparedRequest>,
        ready: std::sync::mpsc::SyncSender<()>,
    ) -> Result<ServingOwnerReport> {
        ensure!(
            config.maximum_request_slots > 0,
            "continuous engine needs request slots"
        );
        ensure!(
            config
                .maximum_request_slots
                .checked_add(config.scheduler.maximum_prefill_tokens)
                .is_some_and(|required| required <= config.maximum_batch_tokens),
            "maximum batch tokens cannot cover decode slots plus prefill chunk"
        );

        // The packed serving weights are additional resident GPU memory. Build
        // them before the physical KV arena and recompute the page budget from
        // the post-pack free-memory state rather than using the earlier config
        // estimate that only accounted for checkpoint-native weights.
        self.model.prepare_packed_qkv_decode(
            &self.runtime,
            config.maximum_request_slots,
            config.maximum_batch_tokens,
        )?;
        let mut config = config;
        let (free_bytes, total_bytes) = self.runtime.memory_info()?;
        let safety_bytes = (total_bytes / 10).max(512 * 1024 * 1024);
        let temporary_pool_budget =
            (64usize * 1024 * 1024 * std::mem::size_of::<bf16>()).saturating_add(32 * 1024 * 1024);
        let available_for_kv = free_bytes
            .saturating_sub(safety_bytes)
            .saturating_sub(temporary_pool_budget);
        let attention_layers = self
            .model
            .config()
            .layer_types
            .iter()
            .filter(|kind| kind.as_str() == "full_attention")
            .count();
        let bytes_per_page = attention_layers
            .checked_mul(2)
            .and_then(|value| value.checked_mul(self.model.config().num_key_value_heads))
            .and_then(|value| value.checked_mul(self.config.kv_page_size.value()))
            .and_then(|value| value.checked_mul(self.model.config().head_dim()))
            .and_then(|value| value.checked_mul(std::mem::size_of::<bf16>()))
            .context("packed QKV KV page byte size overflow")?;
        let maximum_useful_pages = config
            .maximum_request_slots
            .checked_mul(
                config
                    .maximum_sequence_tokens
                    .div_ceil(self.config.kv_page_size.value()),
            )
            .context("packed QKV maximum KV page count overflow")?;
        config.physical_kv_pages = (available_for_kv / bytes_per_page)
            .min(maximum_useful_pages)
            .max(1);

        radix_owner::run_owner_radix(self, config, receiver, ready)
    }
}
