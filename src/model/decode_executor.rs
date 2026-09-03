const CUDA_GRAPH_MIN_CONTEXT_TOKENS: usize = 128;
const CUDA_GRAPH_MAX_CONTEXT_TOKENS_EXCLUSIVE: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecodeGraphBucket {
    Unsplit,
    // Split4,
    Split8,
}

struct DecodeGraphCache {
    enabled: bool,
    unsplit: Option<cudarc::driver::CudaGraph>,
    // split4: Option<cudarc::driver::CudaGraph>,
    split8: Option<cudarc::driver::CudaGraph>,
}

impl DecodeGraphCache {
    fn new(runtime: &CudaRuntime) -> Self {
        Self {
            enabled: runtime.graph_capture_compatible(),
            unsplit: None,
            // split4: None,
            split8: None,
        }
    }

    fn get(&self, bucket: DecodeGraphBucket) -> Option<&cudarc::driver::CudaGraph> {
        match bucket {
            DecodeGraphBucket::Unsplit => self.unsplit.as_ref(),
            // DecodeGraphBucket::Split4 => self.split4.as_ref(),
            DecodeGraphBucket::Split8 => self.split8.as_ref(),
        }
    }

    fn insert(&mut self, bucket: DecodeGraphBucket, graph: cudarc::driver::CudaGraph) {
        match bucket {
            DecodeGraphBucket::Unsplit => self.unsplit = Some(graph),
            // DecodeGraphBucket::Split4 => self.split4 = Some(graph),
            DecodeGraphBucket::Split8 => self.split8 = Some(graph),
        }
    }
}

/// Persistent fixed-address scratch for the single-token-per-segment serving
/// topology. All transformer layers reuse the same buffers sequentially; no
/// workspace is allocated per layer or per decode step. Selective FP8 sites
/// reuse one E4M3 activation scratch buffer across every linear projection.
/// Long-context low-batch attention reuses one bounded FP32 split-K workspace.
pub(crate) struct DecodeExecutor {
    maximum_tokens: usize,
    hidden: Tensor<bf16>,
    normalized: Tensor<bf16>,
    post_operator: Tensor<bf16>,
    operator_output: Tensor<bf16>,
    query: Tensor<bf16>,
    key: Tensor<bf16>,
    value: Tensor<bf16>,
    wide: Tensor<bf16>,
    activated: Tensor<bf16>,
    logits: Tensor<bf16>,
    fp8_input: Tensor<u8>,
    attention_splitk_partials: Tensor<f32>,
    sampled: Tensor<u32>,
    splitk_attention_enabled: bool,
    cuda_graphs: DecodeGraphCache,
}

fn splitk_attention_enabled_from_env() -> bool {
    std::env::var("LFM25_SPLITK_ATTENTION")
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(true)
}

fn linear_decode_prequantized_fp8_into(
    runtime: &CudaRuntime,
    input: &Tensor<u8>,
    fp8: &Fp8LinearWeight,
    output: &mut Tensor<bf16>,
) -> Result<()> {
    ensure!(
        input.rank() == 2,
        "persistent prequantized FP8 linear requires rank-2 input, got {:?}",
        input.dims()
    );
    ensure!(
        fp8.data.rank() == 2,
        "persistent prequantized FP8 weight must be rank 2"
    );
    let m = input.dims()[0];
    let k = input.dims()[1];
    let n = fp8.data.dims()[0];
    ensure!(m > 0 && k > 0 && n > 0, "persistent FP8 linear is empty");
    ensure!(
        fp8.data.dims()[1] == k,
        "persistent prequantized FP8 K mismatch: input K={k}, weight={:?}",
        fp8.data.dims()
    );
    output.set_logical_shape(Shape::new([m, n]))?;
    unsafe {
        runtime.blaslt().linear_fp8_scaled(
            input.storage(),
            fp8.data.storage(),
            output.storage_mut(),
            crate::cuda::Fp8LinearConfig {
                m,
                n,
                k,
                scale_mode: Fp8ScaleMode::Tensorwide,
                output_scale: fp8.activation_scale.dequantize_multiplier
                    * fp8.weight_scale.dequantize_multiplier,
            },
        )?;
    }
    Ok(())
}

fn linear_decode_into(
    runtime: &CudaRuntime,
    input: &Tensor<bf16>,
    weight: &LinearWeight,
    use_fp8: bool,
    fp8_input: &mut Tensor<u8>,
    output: &mut Tensor<bf16>,
) -> Result<()> {
    let Some(fp8) = weight.fp8.as_ref().filter(|_| use_fp8) else {
        return ops::linear_bf16_into(runtime, input, &weight.bf16, output);
    };

    ensure!(
        input.rank() == 2,
        "persistent FP8 decode linear requires rank-2 input, got {:?}",
        input.dims()
    );
    ensure!(
        fp8.data.rank() == 2,
        "persistent FP8 decode weight must be rank 2"
    );
    let m = input.dims()[0];
    let k = input.dims()[1];
    let n = fp8.data.dims()[0];
    ensure!(
        m > 0 && k > 0 && n > 0,
        "persistent FP8 decode linear is empty"
    );
    ensure!(
        fp8.data.dims()[1] == k,
        "persistent FP8 decode K mismatch: input K={k}, weight={:?}",
        fp8.data.dims()
    );

    fp8_input.set_logical_shape(Shape::new([m, k]))?;
    output.set_logical_shape(Shape::new([m, n]))?;
    unsafe {
        runtime.kernels().fp8_quantize().launch_bf16_e4m3(
            runtime.stream(),
            input.storage(),
            fp8_input.storage_mut(),
            input.numel(),
            fp8.activation_scale.quantize_multiplier,
        )?;
    }
    linear_decode_prequantized_fp8_into(runtime, fp8_input, fp8, output)
}

impl DecodeExecutor {
    fn new(runtime: &CudaRuntime, config: &Lfm2Config, maximum_tokens: usize) -> Result<Self> {
        ensure!(maximum_tokens > 0, "decode executor requires token capacity");
        let hidden = config.hidden_size;
        let intermediate = config.effective_intermediate_size();
        let kv_width = config.num_key_value_heads * config.head_dim();
        let wide_width = (intermediate * 2).max(hidden * 3);
        let fp8_input_width = intermediate.max(hidden);
        let splitk_workspace = ops::splitk_workspace_elements(maximum_tokens)?;

        Ok(Self {
            maximum_tokens,
            hidden: runtime.alloc_uninit::<bf16>(Shape::new([maximum_tokens, hidden]))?,
            normalized: runtime.alloc_uninit::<bf16>(Shape::new([maximum_tokens, hidden]))?,
            post_operator: runtime.alloc_uninit::<bf16>(Shape::new([maximum_tokens, hidden]))?,
            operator_output: runtime.alloc_uninit::<bf16>(Shape::new([maximum_tokens, hidden]))?,
            query: runtime.alloc_uninit::<bf16>(Shape::new([maximum_tokens, hidden]))?,
            key: runtime.alloc_uninit::<bf16>(Shape::new([maximum_tokens, kv_width]))?,
            value: runtime.alloc_uninit::<bf16>(Shape::new([maximum_tokens, kv_width]))?,
            wide: runtime.alloc_uninit::<bf16>(Shape::new([maximum_tokens, wide_width]))?,
            activated: runtime.alloc_uninit::<bf16>(Shape::new([maximum_tokens, intermediate]))?,
            logits: runtime.alloc_uninit::<bf16>(Shape::new([maximum_tokens, config.vocab_size]))?,
            fp8_input: runtime.alloc_fp8(Shape::new([maximum_tokens, fp8_input_width]))?,
            attention_splitk_partials: runtime.alloc_uninit::<f32>(Shape::new([
                splitk_workspace,
            ]))?,
            sampled: runtime.alloc_uninit::<u32>(Shape::new([maximum_tokens]))?,
            splitk_attention_enabled: splitk_attention_enabled_from_env(),
            cuda_graphs: DecodeGraphCache::new(runtime),
        })
    }

    #[inline]
    fn eligible(&self, input: &RaggedBatchInput<'_>) -> bool {
        let tokens = input.token_ids.len();
        if tokens == 0 || tokens > self.maximum_tokens {
            return false;
        }
        if input.positions.len() != tokens
            || input.request_slots.len() != tokens
            || input.segment_slots.len() != tokens
            || input.segment_offsets.len() != tokens + 1
            || input.output_rows.len() != tokens
        {
            return false;
        }
        input
            .segment_offsets
            .iter()
            .enumerate()
            .all(|(index, &offset)| usize::try_from(offset).ok() == Some(index))
            && input
                .output_rows
                .iter()
                .enumerate()
                .all(|(index, &row)| usize::try_from(row).ok() == Some(index))
    }

    fn cuda_graph_bucket(&self, cache: &BatchModelCache) -> Option<DecodeGraphBucket> {
        if !self.cuda_graphs.enabled || !self.splitk_attention_enabled {
            return None;
        }
        let metadata = &cache.gpu_batch;
        let num_tokens = metadata.token_ids().numel();
        let context_tokens = metadata.max_context_tokens();
        let page_size = cache.page_size.value();
        if num_tokens != 1
            || page_size != 16
            || !(CUDA_GRAPH_MIN_CONTEXT_TOKENS..CUDA_GRAPH_MAX_CONTEXT_TOKENS_EXCLUSIVE)
                .contains(&context_tokens)
            || ops::should_use_mok_one_kernel(page_size, context_tokens, num_tokens)
        {
            return None;
        }
        match ops::splitk_decode_splits(num_tokens, context_tokens, page_size) {
            1 => Some(DecodeGraphBucket::Unsplit),
            4 => None,
            8 => Some(DecodeGraphBucket::Split8),
            _ => None,
        }
    }

    fn forward_prepared_dispatch(
        &mut self,
        model: &Lfm2Model,
        runtime: &CudaRuntime,
        cache: &mut BatchModelCache,
    ) -> Result<()> {
        let Some(bucket) = self.cuda_graph_bucket(cache) else {
            return self.forward_prepared(model, runtime, cache);
        };

        // Graph replay executes fixed B1 kernel arguments against persistent
        // storage. A prior direct larger-batch step can leave the host-side
        // logical shape of `sampled` larger than one, so restore the externally
        // observed shape before replay.
        self.sampled.set_logical_shape(Shape::new([1]))?;
        if let Some(graph) = self.cuda_graphs.get(bucket) {
            graph
                .launch()
                .context("bounded CUDA Graph decode replay failed")?;
            return Ok(());
        }

        // First use of each measured-safe topology is captured lazily. Metadata
        // preparation has already completed outside capture. Synchronize once so
        // capture cannot inherit an uncaptured dependency, then execute the new
        // graph once because stream capture records work without running it.
        runtime.synchronize()?;
        let stream = runtime.stream();
        stream
            .begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .context("failed to begin bounded CUDA Graph decode capture")?;
        self.forward_prepared(model, runtime, cache)?;
        let graph = stream
            .end_capture(
                cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY,
            )
            .context("failed to end bounded CUDA Graph decode capture")?
            .context("bounded CUDA Graph decode capture returned no graph")?;
        graph
            .upload()
            .context("failed to upload bounded CUDA Graph decode")?;
        runtime.synchronize()?;
        graph
            .launch()
            .context("failed to execute newly captured CUDA Graph decode")?;
        self.cuda_graphs.insert(bucket, graph);
        Ok(())
    }

    #[cfg(test)]
    fn set_cuda_graphs_enabled_for_test(&mut self, enabled: bool) {
        self.cuda_graphs.enabled = enabled;
    }

    fn forward_prepared(
        &mut self,
        model: &Lfm2Model,
        runtime: &CudaRuntime,
        cache: &mut BatchModelCache,
    ) -> Result<()> {
        let BatchModelCache {
            layers, gpu_batch, ..
        } = cache;
        let metadata = &*gpu_batch;
        let num_tokens = metadata.token_ids().numel();
        ensure!(
            num_tokens > 0 && num_tokens <= self.maximum_tokens,
            "decode executor token count exceeds workspace"
        );
        let use_fp8 = model.decode_fp8_enabled;
        let splitk_attention_enabled = self.splitk_attention_enabled;

        let Self {
            hidden,
            normalized,
            post_operator,
            operator_output,
            query,
            key,
            value,
            wide,
            activated,
            logits,
            fp8_input,
            attention_splitk_partials,
            sampled,
            ..
        } = self;

        ops::embedding_bf16_into(runtime, metadata.token_ids(), &model.weights.embedding, hidden)?;
        ops::rms_norm_bf16_into(
            runtime,
            hidden,
            &model.weights.layers[0].operator_norm,
            model.config.norm_eps,
            normalized,
        )?;

        for (layer, layer_cache) in layers.iter_mut().enumerate() {
            let weights = &model.weights.layers[layer];
            match (&weights.operator, layer_cache) {
                (OperatorWeights::Conv(conv), BatchLayerCache::Conv(states)) => {
                    linear_decode_into(runtime, normalized, &conv.input, use_fp8, fp8_input, wide)?;
                    ops::short_conv_segmented_lfm2_bf16_into(
                        runtime,
                        wide,
                        &conv.convolution,
                        states,
                        metadata.segment_offsets(),
                        metadata.segment_slots(),
                        post_operator,
                    )?;
                    linear_decode_into(
                        runtime,
                        post_operator,
                        &conv.output,
                        use_fp8,
                        fp8_input,
                        operator_output,
                    )?;
                }
                (OperatorWeights::Attention(attention), BatchLayerCache::Attention(arena)) => {
                    linear_decode_into(
                        runtime,
                        normalized,
                        &attention.query,
                        use_fp8,
                        fp8_input,
                        query,
                    )?;
                    query.set_logical_shape(Shape::new([num_tokens, 32, 64]))?;
                    linear_decode_into(
                        runtime,
                        normalized,
                        &attention.key,
                        use_fp8,
                        fp8_input,
                        key,
                    )?;
                    key.set_logical_shape(Shape::new([num_tokens, 8, 64]))?;
                    linear_decode_into(
                        runtime,
                        normalized,
                        &attention.value,
                        use_fp8,
                        fp8_input,
                        value,
                    )?;
                    value.set_logical_shape(Shape::new([num_tokens, 8, 64]))?;

                    if ops::should_use_mok_one_kernel(
                        arena.page_size().value(),
                        metadata.max_context_tokens(),
                        num_tokens,
                    ) {
                        ops::fused_ragged_paged_attention_decode_lfm2_bf16_into(
                            runtime,
                            ops::FusedRaggedAttentionInput {
                                attention: ops::FusedAttentionInput {
                                    query_raw: query,
                                    key_raw: key,
                                    value_raw: value,
                                    query_norm: &attention.query_norm,
                                    key_norm: &attention.key_norm,
                                    inv_freq: &model.inv_freq,
                                    position_ids: metadata.positions(),
                                    slot_mapping: metadata.physical_slots(),
                                    eps: model.config.norm_eps,
                                },
                                arena,
                                block_tables: metadata.block_tables(),
                                block_table_stride: metadata.block_table_stride(),
                                request_slots: metadata.request_slots(),
                            },
                            post_operator,
                        )?;
                    } else {
                        ops::qk_norm_rope_kv_write_arena_decode_bf16(
                            runtime,
                            ops::QkPostprocessInput {
                                query,
                                key,
                                value,
                                query_norm: &attention.query_norm,
                                key_norm: &attention.key_norm,
                                inv_freq: &model.inv_freq,
                                position_ids: metadata.positions(),
                                slot_mapping: metadata.physical_slots(),
                                eps: model.config.norm_eps,
                            },
                            arena,
                        )?;
                        let input = ops::FastRaggedAttentionInput {
                            query,
                            arena,
                            block_tables: metadata.block_tables(),
                            block_table_stride: metadata.block_table_stride(),
                            request_slots: metadata.request_slots(),
                            position_ids: metadata.positions(),
                        };
                        let splits = if splitk_attention_enabled {
                            ops::splitk_decode_splits(
                                num_tokens,
                                metadata.max_context_tokens(),
                                arena.page_size().value(),
                            )
                        } else {
                            1
                        };
                        if splits > 1 {
                            ops::paged_ragged_attention_splitk_lfm2_bf16_into(
                                runtime,
                                input,
                                attention_splitk_partials,
                                splits,
                                post_operator,
                            )?;
                        } else {
                            ops::paged_ragged_attention_fast_lfm2_bf16_into(
                                runtime,
                                input,
                                post_operator,
                            )?;
                        }
                    }
                    post_operator
                        .set_logical_shape(Shape::new([num_tokens, model.config.hidden_size]))?;
                    linear_decode_into(
                        runtime,
                        post_operator,
                        &attention.output,
                        use_fp8,
                        fp8_input,
                        operator_output,
                    )?;
                }
                _ => anyhow::bail!("model/batch cache layer type mismatch at layer {layer}"),
            }

            ops::residual_rms_norm_bf16_into(
                runtime,
                hidden,
                operator_output,
                &weights.ffn_norm,
                model.config.norm_eps,
                post_operator,
                normalized,
            )?;

            linear_decode_into(
                runtime,
                normalized,
                &weights.feed_forward.gate_up,
                use_fp8,
                fp8_input,
                wide,
            )?;
            if let Some(fp8) = weights.feed_forward.down.fp8.as_ref().filter(|_| use_fp8) {
                ops::silu_mul_packed_bf16_to_e4m3_into(
                    runtime,
                    wide,
                    fp8_input,
                    fp8.activation_scale.quantize_multiplier,
                )?;
                linear_decode_prequantized_fp8_into(runtime, fp8_input, fp8, operator_output)?;
            } else {
                ops::silu_mul_packed_bf16_into(runtime, wide, activated)?;
                ops::linear_bf16_into(
                    runtime,
                    activated,
                    &weights.feed_forward.down.bf16,
                    operator_output,
                )?;
            }

            let next_norm = if layer + 1 < model.config.num_hidden_layers {
                &model.weights.layers[layer + 1].operator_norm
            } else {
                &model.weights.final_norm
            };
            ops::residual_rms_norm_bf16_into(
                runtime,
                post_operator,
                operator_output,
                next_norm,
                model.config.norm_eps,
                hidden,
                normalized,
            )?;
        }

        // Pure decode has exactly one output row per token, in row order, so
        // the gather stage used by ragged prefill is unnecessary here.
        if let (true, Some(fp8)) = (use_fp8, model.weights.lm_head_fp8.as_ref()) {
            ensure!(
                fp8.data.dims() == model.weights.embedding.dims(),
                "FP8 LM head shape must match tied embedding"
            );
            fp8_input.set_logical_shape(Shape::new([num_tokens, model.config.hidden_size]))?;
            logits.set_logical_shape(Shape::new([num_tokens, model.config.vocab_size]))?;
            unsafe {
                runtime.kernels().fp8_quantize().launch_bf16_e4m3(
                    runtime.stream(),
                    normalized.storage(),
                    fp8_input.storage_mut(),
                    normalized.numel(),
                    fp8.activation_scale.quantize_multiplier,
                )?;
            }
            linear_decode_prequantized_fp8_into(runtime, fp8_input, fp8, logits)?;
        } else {
            ops::linear_bf16_into(runtime, normalized, &model.weights.embedding, logits)?;
        }
        ops::argmax_rows_bf16_into(runtime, logits, sampled)
    }
}

impl Lfm2Model {
    fn prepare_decode_executor_fp8(
        &self,
        runtime: &CudaRuntime,
        maximum_batch: usize,
    ) -> Result<()> {
        if !self.decode_fp8_enabled {
            return Ok(());
        }
        for batch in 1..=maximum_batch {
            for layer in &self.weights.layers {
                for weight in [&layer.feed_forward.gate_up, &layer.feed_forward.down] {
                    if weight.fp8.is_some() {
                        runtime.blaslt().prepare_linear_fp8(
                            batch,
                            weight.bf16.dims()[0],
                            weight.bf16.dims()[1],
                            Fp8ScaleMode::Tensorwide,
                        )?;
                    }
                }
                match &layer.operator {
                    OperatorWeights::Conv(conv) => {
                        for weight in [&conv.input, &conv.output] {
                            if weight.fp8.is_some() {
                                runtime.blaslt().prepare_linear_fp8(
                                    batch,
                                    weight.bf16.dims()[0],
                                    weight.bf16.dims()[1],
                                    Fp8ScaleMode::Tensorwide,
                                )?;
                            }
                        }
                    }
                    OperatorWeights::Attention(attention) => {
                        for weight in [
                            &attention.query,
                            &attention.key,
                            &attention.value,
                            &attention.output,
                        ] {
                            if weight.fp8.is_some() {
                                runtime.blaslt().prepare_linear_fp8(
                                    batch,
                                    weight.bf16.dims()[0],
                                    weight.bf16.dims()[1],
                                    Fp8ScaleMode::Tensorwide,
                                )?;
                            }
                        }
                    }
                }
            }
            if self.weights.lm_head_fp8.is_some() {
                runtime.blaslt().prepare_linear_fp8(
                    batch,
                    self.weights.embedding.dims()[0],
                    self.weights.embedding.dims()[1],
                    Fp8ScaleMode::Tensorwide,
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn new_decode_executor(
        &self,
        runtime: &CudaRuntime,
        maximum_tokens: usize,
    ) -> Result<DecodeExecutor> {
        self.prepare_decode_executor_fp8(runtime, maximum_tokens)?;
        DecodeExecutor::new(runtime, &self.config, maximum_tokens)
    }

    pub(crate) fn try_forward_ragged_decode<'a>(
        &self,
        runtime: &CudaRuntime,
        cache: &mut BatchModelCache,
        executor: &'a mut DecodeExecutor,
        input: &RaggedBatchInput<'_>,
    ) -> Result<Option<&'a Tensor<u32>>> {
        if !executor.eligible(input) {
            return Ok(None);
        }
        cache.prepare_ragged(runtime, input)?;
        executor.forward_prepared_dispatch(self, runtime, cache)?;
        Ok(Some(&executor.sampled))
    }
}
