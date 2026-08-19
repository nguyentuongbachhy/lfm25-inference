use std::cell::RefCell;

struct PackedQkvDecodeWeights {
    model_key: usize,
    layers: Vec<Option<Tensor<bf16>>>,
}

std::thread_local! {
    static PACKED_QKV_DECODE_WEIGHTS: RefCell<Option<PackedQkvDecodeWeights>> = const {
        RefCell::new(None)
    };
}

impl Lfm2Model {
    #[inline]
    fn packed_qkv_model_key(&self) -> usize {
        self.weights.layers.as_ptr() as usize
    }

    /// Prepares persistent BF16 [Q|K|V] projection weights for the serving
    /// owner thread. The sidecar intentionally leaves the checkpoint-native
    /// Q/K/V tensors untouched so prefill, FP8, calibration and the measured
    /// short-context one-kernel decode path remain byte-for-byte on the old
    /// execution route.
    pub(crate) fn prepare_packed_qkv_decode(
        &self,
        runtime: &CudaRuntime,
        maximum_request_slots: usize,
        maximum_batch_tokens: usize,
    ) -> Result<()> {
        ensure!(maximum_request_slots > 0, "packed QKV requires request slots");
        ensure!(maximum_batch_tokens > 0, "packed QKV requires batch capacity");

        let model_key = self.packed_qkv_model_key();
        PACKED_QKV_DECODE_WEIGHTS.with(|storage| -> Result<()> {
            if storage
                .borrow()
                .as_ref()
                .is_some_and(|weights| weights.model_key == model_key)
            {
                return Ok(());
            }

            let mut layers = (0..self.weights.layers.len())
                .map(|_| None)
                .collect::<Vec<_>>();
            for (layer, weights) in self.weights.layers.iter().enumerate() {
                let OperatorWeights::Attention(attention) = &weights.operator else {
                    continue;
                };
                let qk = runtime
                    .pack_rows_bf16(&attention.query.bf16, &attention.key.bf16)
                    .with_context(|| format!("failed to pack Q/K weights for layer {layer}"))?;
                let qkv = runtime
                    .pack_rows_bf16(&qk, &attention.value.bf16)
                    .with_context(|| format!("failed to pack Q/K/V weights for layer {layer}"))?;
                ensure!(
                    qkv.dims() == [3072, self.config.hidden_size],
                    "packed QKV weight has unexpected shape at layer {layer}: {:?}",
                    qkv.dims()
                );
                layers[layer] = Some(qkv);
            }
            *storage.borrow_mut() = Some(PackedQkvDecodeWeights { model_key, layers });
            Ok(())
        })?;

        // Keep cuBLASLt plan creation outside the measured owner loop. The
        // candidate batch set mirrors BatchModelCache::new_batch_cache().
        let mut batches = Vec::with_capacity(maximum_request_slots.saturating_mul(12));
        batches.extend(1..=maximum_request_slots);
        let mut chunk = 1usize;
        while chunk <= maximum_batch_tokens {
            for decode in 0..=maximum_request_slots {
                let Some(batch) = decode.checked_add(chunk) else {
                    continue;
                };
                if batch <= maximum_batch_tokens {
                    batches.push(batch);
                }
            }
            let Some(next) = chunk.checked_mul(2) else {
                break;
            };
            chunk = next;
        }
        batches.sort_unstable();
        batches.dedup();
        for batch in batches {
            runtime
                .blaslt()
                .prepare_linear_bf16(batch, 3072, self.config.hidden_size)?;
        }
        Ok(())
    }

    pub(crate) fn forward_ragged_batch_packed_qkv(
        &self,
        runtime: &CudaRuntime,
        cache: &mut BatchModelCache,
        input: RaggedBatchInput<'_>,
    ) -> Result<Tensor<bf16>> {
        ensure!(!input.token_ids.is_empty(), "ragged batch is empty");
        cache.prepare_ragged(runtime, &input)?;
        let model_key = self.packed_qkv_model_key();
        PACKED_QKV_DECODE_WEIGHTS.with(|storage| {
            let storage = storage.borrow();
            let packed = storage
                .as_ref()
                .context("packed QKV serving weights were not prepared")?;
            ensure!(
                packed.model_key == model_key,
                "packed QKV weights belong to a different model instance"
            );
            ensure!(
                packed.layers.len() == self.weights.layers.len(),
                "packed QKV layer count mismatch"
            );
            self.forward_prepared_batch_packed_qkv(runtime, cache, &packed.layers)
        })
    }

    fn forward_prepared_batch_packed_qkv(
        &self,
        runtime: &CudaRuntime,
        cache: &mut BatchModelCache,
        packed_qkv: &[Option<Tensor<bf16>>],
    ) -> Result<Tensor<bf16>> {
        let token_count = cache.gpu_batch.token_ids().numel();
        let use_fp8 = self.decode_fp8_enabled && token_count <= self.maximum_fp8_batch;
        let metadata = &cache.gpu_batch;
        let mut hidden =
            ops::embedding_bf16(runtime, metadata.token_ids(), &self.weights.embedding)?;
        let mut normalized = ops::rms_norm_bf16(
            runtime,
            &hidden,
            &self.weights.layers[0].operator_norm,
            self.config.norm_eps,
        )?;

        for layer in 0..self.config.num_hidden_layers {
            let weights = &self.weights.layers[layer];
            let operator_output = match (&weights.operator, &mut cache.layers[layer]) {
                (OperatorWeights::Attention(operator), BatchLayerCache::Attention(arena)) => {
                    self.attention_batch_packed_qkv(
                        runtime,
                        operator,
                        packed_qkv[layer].as_ref(),
                        normalized,
                        arena,
                        metadata,
                        use_fp8,
                    )?
                }
                (OperatorWeights::Conv(operator), BatchLayerCache::Conv(states)) => self
                    .short_conv_batch(runtime, operator, &normalized, states, metadata, use_fp8)?,
                _ => anyhow::bail!("model/batch cache layer type mismatch at layer {layer}"),
            };
            let (post_operator, ffn_input) = ops::residual_rms_norm_bf16(
                runtime,
                &hidden,
                &operator_output,
                &weights.ffn_norm,
                self.config.norm_eps,
            )?;
            let ffn_output = self.feed_forward(
                runtime,
                weights,
                &ffn_input,
                LayerExecution {
                    profile: None,
                    calibration: None,
                    layer,
                    use_fp8,
                },
            )?;
            let next_norm = if layer + 1 < self.config.num_hidden_layers {
                &self.weights.layers[layer + 1].operator_norm
            } else {
                &self.weights.final_norm
            };
            (hidden, normalized) = ops::residual_rms_norm_bf16(
                runtime,
                &post_operator,
                &ffn_output,
                next_norm,
                self.config.norm_eps,
            )?;
        }

        let logits_input = ops::gather_rows_bf16(runtime, &normalized, metadata.output_rows())?;
        match (use_fp8, self.weights.lm_head_fp8.as_ref()) {
            (true, Some(fp8)) => ops::linear_fp8_e4m3(
                runtime,
                &logits_input,
                &fp8.data,
                fp8.activation_scale,
                fp8.weight_scale,
            ),
            _ => ops::linear_bf16(runtime, &logits_input, &self.weights.embedding),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_batch_packed_qkv(
        &self,
        runtime: &CudaRuntime,
        weights: &AttentionWeights,
        packed_qkv: Option<&Tensor<bf16>>,
        normalized: Tensor<bf16>,
        arena: &mut PagedKvArena,
        metadata: &GpuBatch,
        use_fp8: bool,
    ) -> Result<Tensor<bf16>> {
        let num_tokens = normalized.dims()[0];
        let decode_only = metadata.segment_slots().numel() == num_tokens
            && metadata.segment_offsets().numel() == num_tokens + 1;
        let one_kernel_decode = decode_only
            && ops::should_use_mok_one_kernel(
                arena.page_size().value(),
                metadata.max_context_tokens(),
                num_tokens,
            );

        // The packed projection is deliberately restricted to the production
        // BF16 two-kernel decode region. Short-context one-kernel decode keeps
        // its measured Q/K/V layout, while FP8 and non-decode work stay on the
        // established path.
        if decode_only && !one_kernel_decode && !use_fp8 {
            let packed_weight = packed_qkv.context("attention layer is missing packed QKV weight")?;
            let projected = ops::linear_bf16(runtime, &normalized, packed_weight)?;
            ensure!(
                projected.dims() == [num_tokens, 3072],
                "packed QKV projection shape mismatch: {:?}",
                projected.dims()
            );
            let query = ops::qk_norm_rope_kv_write_arena_packed_decode_bf16(
                runtime,
                ops::PackedQkvPostprocessInput {
                    packed_qkv: &projected,
                    query_norm: &weights.query_norm,
                    key_norm: &weights.key_norm,
                    inv_freq: &self.inv_freq,
                    position_ids: metadata.positions(),
                    slot_mapping: metadata.physical_slots(),
                    eps: self.config.norm_eps,
                },
                arena,
            )?;
            let attended = ops::paged_ragged_attention_fast_lfm2_bf16(
                runtime,
                ops::FastRaggedAttentionInput {
                    query: &query,
                    arena,
                    block_tables: metadata.block_tables(),
                    block_table_stride: metadata.block_table_stride(),
                    request_slots: metadata.request_slots(),
                    position_ids: metadata.positions(),
                },
            )?
            .reshape(Shape::new([num_tokens, self.config.hidden_size]))?;
            return linear_dispatch(runtime, &attended, &weights.output, false);
        }

        self.attention_batch(runtime, weights, normalized, arena, metadata, use_fp8)
    }
}
