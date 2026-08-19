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
        crate::tensor::enter_buffer_pool_owner_mode();
        radix_owner::run_owner_radix(self, config, receiver, ready)
    }
}
