use cudarc::driver::{
    CudaGraph,
    sys::{CUgraphInstantiate_flags, CUstreamCaptureMode},
};

const MAX_DECODE_GRAPHS: usize = 32;
const DECODE_GRAPH_INSTANTIATE_FLAGS: CUgraphInstantiate_flags =
    CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum DecodeGraphPath {
    OneKernel,
    TwoKernel,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct DecodeGraphKey {
    tokens: usize,
    path: DecodeGraphPath,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DecodeGraphStats {
    pub(crate) entries: usize,
    pub(crate) captures: u64,
    pub(crate) replays: u64,
    pub(crate) capture_failures: u64,
    pub(crate) direct_steps: u64,
}

/// Persistent fixed-address scratch for the BF16 single-token-per-segment
/// serving topology. All transformer layers reuse the same buffers
/// sequentially; no workspace is allocated per layer or per decode step.
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
    sampled: Tensor<u32>,
    graphs: HashMap<DecodeGraphKey, CudaGraph>,
    seen_graphs: HashMap<DecodeGraphKey, ()>,
    failed_graphs: HashMap<DecodeGraphKey, ()>,
    graph_captures: u64,
    graph_replays: u64,
    graph_capture_failures: u64,
    direct_steps: u64,
}

impl DecodeExecutor {
    fn new(runtime: &CudaRuntime, config: &Lfm2Config, maximum_tokens: usize) -> Result<Self> {
        ensure!(maximum_tokens > 0, "decode executor requires token capacity");
        let hidden = config.hidden_size;
        let intermediate = config.effective_intermediate_size();
        let kv_width = config.num_key_value_heads * config.head_dim();
        let wide_width = (intermediate * 2).max(hidden * 3);

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
            sampled: runtime.alloc_uninit::<u32>(Shape::new([maximum_tokens]))?,
            graphs: HashMap::with_capacity(MAX_DECODE_GRAPHS),
            seen_graphs: HashMap::with_capacity(MAX_DECODE_GRAPHS),
            failed_graphs: HashMap::new(),
            graph_captures: 0,
            graph_replays: 0,
            graph_capture_failures: 0,
            direct_steps: 0,
        })
    }

    #[inline]
    fn eligible(&self, model: &Lfm2Model, input: &RaggedBatchInput<'_>) -> bool {
        let tokens = input.token_ids.len();
        if tokens == 0 || tokens > self.maximum_tokens {
            return false;
        }
        let use_fp8 = model.decode_fp8_enabled && tokens <= model.maximum_fp8_batch;
        if use_fp8
            || input.positions.len() != tokens
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

    #[inline]
    fn graph_key(&self, cache: &BatchModelCache) -> DecodeGraphKey {
        let tokens = cache.gpu_batch.token_ids().numel();
        let one_kernel = ops::should_use_mok_one_kernel(
            cache.page_size.value(),
            cache.gpu_batch.max_context_tokens(),
            tokens,
        );
        DecodeGraphKey {
            tokens,
            path: if one_kernel {
                DecodeGraphPath::OneKernel
            } else {
                DecodeGraphPath::TwoKernel
            },
        }
    }

    fn forward_prepared_graph(
        &mut self,
        model: &Lfm2Model,
        runtime: &CudaRuntime,
        cache: &mut BatchModelCache,
    ) -> Result<()> {
        let key = self.graph_key(cache);
        let tokens = key.tokens;

        // Graph replay executes only GPU nodes. Keep the host-side logical shape
        // in sync so the following D2H copies exactly the active sampled rows.
        self.sampled.set_logical_shape(Shape::new([tokens]))?;

        if let Some(graph) = self.graphs.get(&key) {
            graph
                .launch()
                .context("failed to launch cached decode CUDA graph")?;
            self.graph_replays = self.graph_replays.saturating_add(1);
            return Ok(());
        }

        // Do not pay graph instantiation on the first observation of a topology.
        // This keeps the first-token/tail-prefill path direct and only captures
        // shapes that demonstrably recur.
        if self.seen_graphs.insert(key, ()).is_none() {
            self.direct_steps = self.direct_steps.saturating_add(1);
            return self.forward_prepared(model, runtime, cache);
        }

        if self.graphs.len() >= MAX_DECODE_GRAPHS || self.failed_graphs.contains_key(&key) {
            self.direct_steps = self.direct_steps.saturating_add(1);
            return self.forward_prepared(model, runtime, cache);
        }

        if let Err(error) = runtime
            .stream()
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        {
            eprintln!("decode graph capture failed: key={key:?} stage=begin_capture error={error}");
            self.graph_capture_failures = self.graph_capture_failures.saturating_add(1);
            self.failed_graphs.insert(key, ());
            self.direct_steps = self.direct_steps.saturating_add(1);
            return self.forward_prepared(model, runtime, cache);
        }

        if let Err(error) = self.forward_prepared(model, runtime, cache) {
            // Capture-specific failures must not take down serving. Terminate the
            // capture, blacklist this topology, and submit the normal path once.
            // If the model itself is invalid, that direct retry returns the real
            // error to the caller.
            let end_capture_error = runtime
                .stream()
                .end_capture(DECODE_GRAPH_INSTANTIATE_FLAGS)
                .err();
            eprintln!("decode graph capture failed: key={key:?} stage=forward error={error}");
            if let Some(end_error) = end_capture_error {
                eprintln!(
                    "decode graph capture cleanup failed: key={key:?} stage=end_capture error={end_error}"
                );
            }
            self.graph_capture_failures = self.graph_capture_failures.saturating_add(1);
            self.failed_graphs.insert(key, ());
            self.direct_steps = self.direct_steps.saturating_add(1);
            return self.forward_prepared(model, runtime, cache);
        }

        let graph = match runtime
            .stream()
            .end_capture(DECODE_GRAPH_INSTANTIATE_FLAGS)
        {
            Ok(Some(graph)) => graph,
            Ok(None) => {
                eprintln!("decode graph capture failed: key={key:?} stage=end_capture error=no_graph");
                self.graph_capture_failures = self.graph_capture_failures.saturating_add(1);
                self.failed_graphs.insert(key, ());
                self.direct_steps = self.direct_steps.saturating_add(1);
                // Work submitted while capturing did not execute. Re-submit the
                // normal executor path so this serving step still completes.
                return self.forward_prepared(model, runtime, cache);
            }
            Err(error) => {
                eprintln!("decode graph capture failed: key={key:?} stage=end_capture error={error}");
                self.graph_capture_failures = self.graph_capture_failures.saturating_add(1);
                self.failed_graphs.insert(key, ());
                self.direct_steps = self.direct_steps.saturating_add(1);
                // Work submitted while capturing did not execute. Re-submit the
                // normal executor path so this serving step still completes.
                return self.forward_prepared(model, runtime, cache);
            }
        };

        // `end_capture` uses cuGraphInstantiate, where the UPLOAD instantiate
        // flag is invalid. Pre-upload explicitly through the graph API instead.
        if let Err(error) = graph.upload() {
            eprintln!("decode graph upload failed: key={key:?} error={error}");
        }

        // Captured operations have not executed yet. Launch once for the current
        // step, then retain the executable graph for subsequent matching steps.
        graph
            .launch()
            .context("failed to launch newly captured decode CUDA graph")?;
        self.graph_captures = self.graph_captures.saturating_add(1);
        self.graphs.insert(key, graph);
        Ok(())
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
                    ops::linear_bf16_into(runtime, normalized, &conv.input.bf16, wide)?;
                    ops::short_conv_segmented_lfm2_bf16_into(
                        runtime,
                        wide,
                        &conv.convolution,
                        states,
                        metadata.segment_offsets(),
                        metadata.segment_slots(),
                        post_operator,
                    )?;
                    ops::linear_bf16_into(
                        runtime,
                        post_operator,
                        &conv.output.bf16,
                        operator_output,
                    )?;
                }
                (OperatorWeights::Attention(attention), BatchLayerCache::Attention(arena)) => {
                    ops::linear_bf16_into(runtime, normalized, &attention.query.bf16, query)?;
                    query.set_logical_shape(Shape::new([num_tokens, 32, 64]))?;
                    ops::linear_bf16_into(runtime, normalized, &attention.key.bf16, key)?;
                    key.set_logical_shape(Shape::new([num_tokens, 8, 64]))?;
                    ops::linear_bf16_into(runtime, normalized, &attention.value.bf16, value)?;
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
                        ops::paged_ragged_attention_fast_lfm2_bf16_into(
                            runtime,
                            ops::FastRaggedAttentionInput {
                                query,
                                arena,
                                block_tables: metadata.block_tables(),
                                block_table_stride: metadata.block_table_stride(),
                                request_slots: metadata.request_slots(),
                                position_ids: metadata.positions(),
                            },
                            post_operator,
                        )?;
                    }
                    post_operator
                        .set_logical_shape(Shape::new([num_tokens, model.config.hidden_size]))?;
                    ops::linear_bf16_into(
                        runtime,
                        post_operator,
                        &attention.output.bf16,
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

            ops::linear_bf16_into(
                runtime,
                normalized,
                &weights.feed_forward.gate_up.bf16,
                wide,
            )?;
            ops::silu_mul_packed_bf16_into(runtime, wide, activated)?;
            ops::linear_bf16_into(
                runtime,
                activated,
                &weights.feed_forward.down.bf16,
                operator_output,
            )?;

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
        ops::linear_bf16_into(runtime, normalized, &model.weights.embedding, logits)?;
        ops::argmax_rows_bf16_into(runtime, logits, sampled)
    }

    pub(crate) fn graph_stats(&self) -> DecodeGraphStats {
        DecodeGraphStats {
            entries: self.graphs.len(),
            captures: self.graph_captures,
            replays: self.graph_replays,
            capture_failures: self.graph_capture_failures,
            direct_steps: self.direct_steps,
        }
    }
}

impl Lfm2Model {
    pub(crate) fn new_decode_executor(
        &self,
        runtime: &CudaRuntime,
        maximum_tokens: usize,
    ) -> Result<DecodeExecutor> {
        DecodeExecutor::new(runtime, &self.config, maximum_tokens)
    }

    pub(crate) fn try_forward_ragged_decode<'a>(
        &self,
        runtime: &CudaRuntime,
        cache: &mut BatchModelCache,
        executor: &'a mut DecodeExecutor,
        input: &RaggedBatchInput<'_>,
    ) -> Result<Option<&'a Tensor<u32>>> {
        if !executor.eligible(self, input) {
            return Ok(None);
        }
        cache.prepare_ragged(runtime, input)?;
        executor.forward_prepared_graph(self, runtime, cache)?;
        Ok(Some(&executor.sampled))
    }
}
