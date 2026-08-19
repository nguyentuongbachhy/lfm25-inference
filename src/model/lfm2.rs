include!("lfm2_base.rs");

impl BatchModelCache {
    /// Reserve only physical pages that may need to be allocated privately by
    /// this request. Immutable prefix pages already pinned by the radix cache
    /// are not charged again.
    pub(crate) fn reserve_private_pages(
        &mut self,
        request_slot: usize,
        pages: usize,
    ) -> Result<()> {
        ensure!(
            request_slot < self.reservations.len(),
            "request slot out of range"
        );
        ensure!(
            self.reservations[request_slot] == 0,
            "request slot is already reserved"
        );
        ensure!(pages > 0, "request must reserve at least one private KV page");
        let pages = self
            .allocator
            .try_reserve_pages(pages)
            .map_err(anyhow::Error::new)?;
        self.reservations[request_slot] = pages;
        Ok(())
    }

    /// Attach immutable prefix pages to an empty request slot and restore the
    /// recurrent convolution state captured at the same token boundary.
    pub(crate) fn attach_prefix(
        &mut self,
        runtime: &CudaRuntime,
        request_slot: usize,
        prefix_tokens: usize,
        physical_pages: &[u32],
        checkpoints: &super::ConvCheckpointPool,
        checkpoint_slot: u32,
    ) -> Result<()> {
        ensure!(prefix_tokens > 0, "attached prefix cannot be empty");
        ensure!(
            prefix_tokens.is_multiple_of(self.page_size.value()),
            "attached prefix must be page aligned"
        );
        ensure!(
            physical_pages.len() == prefix_tokens / self.page_size.value(),
            "attached prefix page count mismatch"
        );
        ensure!(
            request_slot < self.allocated_tokens.len(),
            "request slot out of range"
        );
        ensure!(
            self.allocated_tokens[request_slot] == 0,
            "prefix can only attach to an empty request slot"
        );
        ensure!(
            self.reservations[request_slot] > 0,
            "request slot has no private KV reservation"
        );

        self.allocator
            .retain_pages(physical_pages)
            .map_err(anyhow::Error::new)?;
        {
            let table = self.block_tables.slot_mut(request_slot)?;
            ensure!(
                table[..physical_pages.len()]
                    .iter()
                    .all(|page| *page == u32::MAX),
                "prefix destination block table is not empty"
            );
            table[..physical_pages.len()].copy_from_slice(physical_pages);
        }
        // From this point onward the normal `release()` path can roll back all
        // request references if a metadata or recurrent-state copy fails.
        self.allocated_tokens[request_slot] = prefix_tokens;
        self.gpu_batch.update_block_table_range(
            runtime,
            request_slot,
            0,
            physical_pages,
        )?;

        let mut convolution_index = 0usize;
        for layer in &mut self.layers {
            if let BatchLayerCache::Conv(states) = layer {
                checkpoints.restore_layer(
                    runtime,
                    checkpoint_slot,
                    convolution_index,
                    states,
                    request_slot,
                )?;
                convolution_index += 1;
            }
        }
        Ok(())
    }

    /// Publish a page-aligned, already-computed prefix. The radix node owns one
    /// reference to every newly introduced physical page, while the checkpoint
    /// pool stores the convolution states at exactly the same boundary.
    pub(crate) fn publish_prefix_checkpoint(
        &mut self,
        runtime: &CudaRuntime,
        request_slot: usize,
        tokens: &[u32],
        prefix_tokens: usize,
        radix: &mut crate::cache::PageRadixCache,
        checkpoints: &mut super::ConvCheckpointPool,
    ) -> Result<bool> {
        if prefix_tokens == 0
            || prefix_tokens > tokens.len()
            || !prefix_tokens.is_multiple_of(self.page_size.value())
            || !radix.can_publish(tokens, prefix_tokens)
        {
            return Ok(false);
        }
        ensure!(
            request_slot < self.allocated_tokens.len(),
            "request slot out of range"
        );
        ensure!(
            self.allocated_tokens[request_slot] >= prefix_tokens,
            "cannot publish prefix beyond allocated KV"
        );
        let Some(checkpoint_slot) = checkpoints.acquire() else {
            return Ok(false);
        };

        let page_count = prefix_tokens / self.page_size.value();
        let physical_pages = self.block_tables.slot(request_slot)?[..page_count].to_vec();
        ensure!(
            physical_pages.iter().all(|page| *page != u32::MAX),
            "published prefix contains unallocated KV pages"
        );

        let mut convolution_index = 0usize;
        for layer in &self.layers {
            if let BatchLayerCache::Conv(states) = layer {
                if let Err(error) = checkpoints.capture_layer(
                    runtime,
                    checkpoint_slot,
                    convolution_index,
                    states,
                    request_slot,
                ) {
                    checkpoints.release(checkpoint_slot);
                    return Err(error);
                }
                convolution_index += 1;
            }
        }

        let newly_cached = match radix.insert_checkpoint(
            tokens,
            prefix_tokens,
            &physical_pages,
            checkpoint_slot,
        ) {
            Ok(pages) => pages,
            Err(error) => {
                checkpoints.release(checkpoint_slot);
                return Err(error);
            }
        };
        self.allocator
            .pin_cached_pages(&newly_cached)
            .map_err(anyhow::Error::new)?;
        Ok(true)
    }
}
