use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cuda::CudaRuntime,
    ops,
    tensor::{Shape, Tensor},
};

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
        input: RaggedBatchInput<'_>,
    ) -> Result<Option<&'a Tensor<u32>>> {
        if !executor.eligible(self, &input) {
            return Ok(None);
        }
        cache.prepare_ragged(runtime, &input)?;
        executor.forward_prepared(self, runtime, cache)?;
        Ok(Some(&executor.sampled))
    }
}
