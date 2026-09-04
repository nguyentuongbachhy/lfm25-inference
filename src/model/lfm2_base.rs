use std::{collections::HashMap, path::Path};

use anyhow::{Context as _, Result, ensure};
use half::bf16;

use crate::{
    cache::{
        FixedBlockTables, KvPageAllocator, KvPageSize, KvPoolSnapshot, PagedKvArena, PagedKvCache,
    },
    config::Lfm2Config,
    cuda::{CudaRuntime, Fp8ScaleMode},
    ops,
    scheduler::{GpuBatch, TransferCounters},
    tensor::{Shape, Tensor},
    weights::WeightStore,
};

use super::{
    CalibrationCollector, CalibrationTensorKind, DecodeProfileMode, Fp8GemmErrorReport,
    HiddenCapture, ModelProfileRecorder, ProfileRegion, characterize_gemm_site, profiled,
    quantization::{Fp8LinearWeight, Fp8PrecisionPolicy, Fp8SitePolicy},
};

enum LayerCache {
    Conv(Tensor<bf16>),
    Attention(PagedKvCache),
}

struct FeedForwardWeights {
    gate_up: LinearWeight,
    down: LinearWeight,
}

struct ConvWeights {
    input: LinearWeight,
    convolution: Tensor<bf16>,
    output: LinearWeight,
}

struct AttentionWeights {
    query: LinearWeight,
    key: LinearWeight,
    value: LinearWeight,
    output: LinearWeight,
    query_norm: Tensor<bf16>,
    key_norm: Tensor<bf16>,
}

struct LinearWeight {
    bf16: Tensor<bf16>,
    fp8: Option<Fp8LinearWeight>,
}

impl LinearWeight {
    fn bf16(weight: Tensor<bf16>) -> Self {
        Self {
            bf16: weight,
            fp8: None,
        }
    }
}

enum OperatorWeights {
    Conv(Box<ConvWeights>),
    Attention(Box<AttentionWeights>),
}

struct LayerWeights {
    operator_norm: Tensor<bf16>,
    operator: OperatorWeights,
    ffn_norm: Tensor<bf16>,
    feed_forward: FeedForwardWeights,
}

struct Lfm2Weights {
    embedding: Tensor<bf16>,
    lm_head_fp8: Option<Fp8LinearWeight>,
    layers: Vec<LayerWeights>,
    final_norm: Tensor<bf16>,
}

impl Lfm2Weights {
    fn from_store(
        runtime: &CudaRuntime,
        config: &Lfm2Config,
        store: &mut WeightStore,
    ) -> Result<Self> {
        let embedding = store.take("model.embed_tokens.weight")?;
        let final_norm = store.take("model.embedding_norm.weight")?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);

        for layer in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            let operator_norm = store.take(&format!("{prefix}.operator_norm.weight"))?;
            let ffn_norm = store.take(&format!("{prefix}.ffn_norm.weight"))?;
            let gate = store.take(&format!("{prefix}.feed_forward.w1.weight"))?;
            let up = store.take(&format!("{prefix}.feed_forward.w3.weight"))?;
            let feed_forward = FeedForwardWeights {
                gate_up: LinearWeight::bf16(runtime.pack_rows_bf16(&gate, &up).with_context(
                    || format!("failed to pack gate/up weights for layer {layer}"),
                )?),
                down: LinearWeight::bf16(store.take(&format!("{prefix}.feed_forward.w2.weight"))?),
            };
            let operator = if config.is_attention_layer(layer) {
                OperatorWeights::Attention(Box::new(AttentionWeights {
                    query: LinearWeight::bf16(
                        store.take(&format!("{prefix}.self_attn.q_proj.weight"))?,
                    ),
                    key: LinearWeight::bf16(
                        store.take(&format!("{prefix}.self_attn.k_proj.weight"))?,
                    ),
                    value: LinearWeight::bf16(
                        store.take(&format!("{prefix}.self_attn.v_proj.weight"))?,
                    ),
                    output: LinearWeight::bf16(
                        store.take(&format!("{prefix}.self_attn.out_proj.weight"))?,
                    ),
                    query_norm: store.take(&format!("{prefix}.self_attn.q_layernorm.weight"))?,
                    key_norm: store.take(&format!("{prefix}.self_attn.k_layernorm.weight"))?,
                }))
            } else {
                OperatorWeights::Conv(Box::new(ConvWeights {
                    input: LinearWeight::bf16(
                        store.take(&format!("{prefix}.conv.in_proj.weight"))?,
                    ),
                    convolution: store.take(&format!("{prefix}.conv.conv.weight"))?,
                    output: LinearWeight::bf16(
                        store.take(&format!("{prefix}.conv.out_proj.weight"))?,
                    ),
                }))
            };
            layers.push(LayerWeights {
                operator_norm,
                operator,
                ffn_norm,
                feed_forward,
            });
        }

        ensure!(
            store.is_empty(),
            "checkpoint contains {} unconsumed tensors",
            store.len()
        );
        Ok(Self {
            embedding,
            lm_head_fp8: None,
            layers,
            final_norm,
        })
    }
}

pub struct SpeculativeCheckpoint {
    pub(crate) start_sequence_length: usize,
    pub(crate) num_tokens: usize,
    pub(crate) conv_histories: Vec<(usize, Tensor<bf16>)>,
}

pub struct ModelCache {
    layers: Vec<LayerCache>,
    sequence_length: usize,
    capacity: usize,
}

enum BatchLayerCache {
    Conv(Tensor<bf16>),
    Attention(PagedKvArena),
}

pub(crate) struct RaggedBatchInput<'a> {
    pub(crate) token_ids: &'a [u32],
    pub(crate) positions: &'a [u32],
    pub(crate) request_slots: &'a [u32],
    pub(crate) segment_offsets: &'a [u32],
    pub(crate) segment_slots: &'a [u32],
    pub(crate) output_rows: &'a [u32],
}

struct LayerExecution<'a> {
    profile: Option<&'a mut ModelProfileRecorder>,
    calibration: Option<&'a mut CalibrationCollector>,
    layer: usize,
    use_fp8: bool,
}

struct AttentionStep<'a> {
    cache: &'a mut PagedKvCache,
    slots: &'a Tensor<i64>,
    positions: &'a Tensor<u32>,
    contiguous_prefill: bool,
    context_tokens: usize,
}

/// Shared serving cache. It owns one physical KV arena per attention layer,
/// while all layers use the same physical page IDs and logical block tables.
pub struct BatchModelCache {
    layers: Vec<BatchLayerCache>,
    allocator: KvPageAllocator,
    block_tables: FixedBlockTables,
    gpu_batch: GpuBatch,
    allocated_tokens: Vec<usize>,
    reservations: Vec<usize>,
    physical_slots_host: Vec<i64>,
    segment_offsets_host: Vec<u32>,
    segment_slots_host: Vec<u32>,
    output_rows_host: Vec<u32>,
    page_size: KvPageSize,
    is_contiguous_prefill: bool,
    is_segmented_prefill: bool,
    max_segment_tokens: usize,
}

impl BatchModelCache {
    fn new(
        runtime: &CudaRuntime,
        config: &Lfm2Config,
        request_slots: usize,
        maximum_batch_tokens: usize,
        physical_pages: usize,
        page_size: KvPageSize,
    ) -> Result<Self> {
        ensure!(request_slots > 0, "batch cache requires request slots");
        ensure!(
            maximum_batch_tokens > 0,
            "batch cache requires token capacity"
        );
        let pages_per_slot = config.max_position_embeddings.div_ceil(page_size.value());
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            if config.is_attention_layer(layer) {
                layers.push(BatchLayerCache::Attention(PagedKvArena::new(
                    runtime,
                    physical_pages,
                    page_size,
                )?));
            } else {
                layers.push(BatchLayerCache::Conv(runtime.zeros::<bf16>(Shape::new(
                    [request_slots, config.hidden_size, config.conv_l_cache - 1],
                ))?));
            }
        }
        Ok(Self {
            layers,
            allocator: KvPageAllocator::new(physical_pages, page_size)?,
            block_tables: FixedBlockTables::new(request_slots, pages_per_slot)?,
            gpu_batch: GpuBatch::new(runtime, maximum_batch_tokens, request_slots, pages_per_slot)?,
            allocated_tokens: vec![0; request_slots],
            reservations: vec![0; request_slots],
            physical_slots_host: Vec::with_capacity(maximum_batch_tokens),
            segment_offsets_host: Vec::with_capacity(request_slots + 1),
            segment_slots_host: Vec::with_capacity(request_slots),
            output_rows_host: Vec::with_capacity(request_slots),
            page_size,
            is_contiguous_prefill: false,
            is_segmented_prefill: false,
            max_segment_tokens: 0,
        })
    }

    pub fn reserve(&mut self, request_slot: usize, maximum_tokens: usize) -> Result<()> {
        ensure!(
            request_slot < self.reservations.len(),
            "request slot out of range"
        );
        ensure!(
            self.reservations[request_slot] == 0,
            "request slot is already reserved"
        );
        let pages = self
            .allocator
            .try_reserve_tokens(maximum_tokens)
            .map_err(anyhow::Error::new)?;
        self.reservations[request_slot] = pages;
        Ok(())
    }

    pub fn release(&mut self, runtime: &CudaRuntime, request_slot: usize) -> Result<()> {
        ensure!(
            request_slot < self.reservations.len(),
            "request slot out of range"
        );
        let tokens = self.allocated_tokens[request_slot];
        let table = self.block_tables.slot_mut(request_slot)?;
        self.allocator.release_sequence(tokens, table);
        self.allocator
            .release_reservation(self.reservations[request_slot]);
        self.reservations[request_slot] = 0;
        self.allocated_tokens[request_slot] = 0;
        for layer in &mut self.layers {
            if let BatchLayerCache::Conv(states) = layer {
                let elements = states.dims()[1]
                    .checked_mul(states.dims()[2])
                    .context("convolution state slot overflow")?;
                let start = request_slot
                    .checked_mul(elements)
                    .context("convolution state offset overflow")?;
                runtime.zero_bf16_range(states, start, elements)?;
            }
        }
        Ok(())
    }

    pub(crate) fn prime_context(
        &mut self,
        runtime: &CudaRuntime,
        request_slots: usize,
        context_tokens: usize,
    ) -> Result<()> {
        ensure!(context_tokens > 0, "primed context must be positive");
        ensure!(
            request_slots <= self.allocated_tokens.len(),
            "primed request count exceeds cache slots"
        );
        for slot in 0..request_slots {
            ensure!(
                self.allocated_tokens[slot] == 0 && self.reservations[slot] > 0,
                "primed slot must be reserved and empty"
            );
            let table = self.block_tables.slot_mut(slot)?;
            let allocated = self
                .allocator
                .grow_sequence(0, context_tokens, table)
                .map_err(anyhow::Error::new)?;
            let pages = context_tokens.div_ceil(self.page_size.value());
            ensure!(allocated == pages, "primed KV page count mismatch");
            self.gpu_batch
                .update_block_table_range(runtime, slot, 0, &table[..pages])?;
            self.allocated_tokens[slot] = context_tokens;
        }
        Ok(())
    }

    fn prepare_tokens(
        &mut self,
        runtime: &CudaRuntime,
        token_ids: &[u32],
        positions: &[u32],
        request_slots: &[u32],
    ) -> Result<()> {
        ensure!(
            token_ids.len() == positions.len(),
            "batch token/position mismatch"
        );
        ensure!(
            token_ids.len() == request_slots.len(),
            "batch token/slot mismatch"
        );
        self.physical_slots_host.clear();
        for (&position, &request_slot) in positions.iter().zip(request_slots) {
            let slot = usize::try_from(request_slot).context("request slot exceeds usize")?;
            ensure!(
                slot < self.allocated_tokens.len(),
                "request slot out of range"
            );
            ensure!(
                self.reservations[slot] > 0,
                "request slot has no KV reservation"
            );
            let target = usize::try_from(position)
                .context("position exceeds usize")?
                .checked_add(1)
                .context("position overflow")?;
            ensure!(
                target == self.allocated_tokens[slot].saturating_add(1),
                "decode positions for slot {slot} must be sequential"
            );
            let table = self.block_tables.slot_mut(slot)?;
            let old_pages = self.allocated_tokens[slot].div_ceil(self.page_size.value());
            let allocated = self
                .allocator
                .grow_sequence(self.allocated_tokens[slot], target, table)
                .map_err(anyhow::Error::new)?;
            if allocated > 0 {
                let new_pages = target.div_ceil(self.page_size.value());
                self.gpu_batch.update_block_table_range(
                    runtime,
                    slot,
                    old_pages,
                    &table[old_pages..new_pages],
                )?;
            }
            let logical_page = usize::try_from(position)? / self.page_size.value();
            let offset = usize::try_from(position)? % self.page_size.value();
            let physical_page = usize::try_from(table[logical_page])?;
            let physical_slot = physical_page
                .checked_mul(self.page_size.value())
                .and_then(|value| value.checked_add(offset))
                .context("physical KV slot overflow")?;
            self.physical_slots_host.push(i64::try_from(physical_slot)?);
            self.allocated_tokens[slot] = target;
        }
        self.gpu_batch.update_step(
            runtime,
            token_ids,
            positions,
            request_slots,
            &self.physical_slots_host,
        )
    }

    fn prepare_decode(
        &mut self,
        runtime: &CudaRuntime,
        token_ids: &[u32],
        positions: &[u32],
        request_slots: &[u32],
    ) -> Result<()> {
        self.is_contiguous_prefill = false;
        self.is_segmented_prefill = false;
        self.max_segment_tokens = 0;
        self.prepare_tokens(runtime, token_ids, positions, request_slots)?;
        self.segment_offsets_host.clear();
        self.segment_slots_host.clear();
        self.output_rows_host.clear();
        self.segment_offsets_host.push(0);
        for (row, &slot) in request_slots.iter().enumerate() {
            self.segment_offsets_host.push(u32::try_from(row + 1)?);
            self.segment_slots_host.push(slot);
            self.output_rows_host.push(u32::try_from(row)?);
        }
        self.gpu_batch.update_segments(
            runtime,
            &self.segment_offsets_host,
            &self.segment_slots_host,
            &self.output_rows_host,
        )
    }

    fn prepare_ragged(
        &mut self,
        runtime: &CudaRuntime,
        input: &RaggedBatchInput<'_>,
    ) -> Result<()> {
        self.is_contiguous_prefill = input.segment_offsets.len() == 2
            && input.positions.first().copied() == Some(0)
            && input.token_ids.len() > 1;
        let all_start_at_zero = input.segment_offsets.windows(2).all(|w| {
            let start = w[0] as usize;
            start < input.positions.len() && input.positions[start] == 0
        });
        self.is_segmented_prefill = input.segment_offsets.len() > 2
            && all_start_at_zero
            && input.token_ids.len() > 1;
        self.max_segment_tokens = if self.is_segmented_prefill {
            input
                .segment_offsets
                .windows(2)
                .map(|w| (w[1] - w[0]) as usize)
                .max()
                .unwrap_or(0)
        } else {
            0
        };
        ensure!(
            input.segment_offsets.last().copied() == Some(u32::try_from(input.token_ids.len())?),
            "last segment offset must equal flattened token count"
        );
        for window in input.segment_offsets.windows(2) {
            ensure!(
                window[0] < window[1],
                "segments must be non-empty and ordered"
            );
        }
        for (&row, &slot) in input.output_rows.iter().zip(input.segment_slots) {
            let row = usize::try_from(row)?;
            ensure!(row < input.token_ids.len(), "output row out of range");
            ensure!(
                input.request_slots[row] == slot,
                "output row does not belong to segment slot"
            );
        }
        self.prepare_tokens(runtime, input.token_ids, input.positions, input.request_slots)?;
        self.gpu_batch.update_segments(
            runtime,
            input.segment_offsets,
            input.segment_slots,
            input.output_rows,
        )
    }

    pub fn kv_snapshot(&self) -> KvPoolSnapshot {
        self.allocator.snapshot()
    }

    pub fn transfers(&self) -> TransferCounters {
        self.gpu_batch.transfers()
    }

    pub(crate) fn begin_serving_measurement(&mut self) {
        self.gpu_batch.reset_transfers();
        self.allocator.reset_peak();
    }
}

impl ModelCache {
    pub fn new(
        runtime: &CudaRuntime,
        config: &Lfm2Config,
        capacity: usize,
        page_size: KvPageSize,
    ) -> Result<Self> {
        ensure!(capacity > 0, "model cache capacity must be positive");
        ensure!(
            capacity <= config.max_position_embeddings,
            "cache capacity {capacity} exceeds model limit {}",
            config.max_position_embeddings
        );
        let mut layers = Vec::with_capacity(config.num_hidden_layers);

        for layer in 0..config.num_hidden_layers {
            if config.is_attention_layer(layer) {
                layers.push(LayerCache::Attention(PagedKvCache::new(
                    runtime, capacity, page_size,
                )?));
            } else {
                layers.push(LayerCache::Conv(runtime.zeros::<bf16>(Shape::new([
                    config.hidden_size,
                    config.conv_l_cache - 1,
                ]))?));
            }
        }

        Ok(Self {
            layers,
            sequence_length: 0,
            capacity,
        })
    }

    #[allow(dead_code)]
    pub fn reset(&mut self, runtime: &CudaRuntime) -> Result<()> {
        self.sequence_length = 0;
        for layer in &mut self.layers {
            if let LayerCache::Conv(state) = layer {
                runtime.zero_bf16_range(state, 0, state.numel())?;
            }
        }
        Ok(())
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }
}

pub struct Lfm2Model {
    config: Lfm2Config,
    weights: Lfm2Weights,
    inv_freq: Tensor<f32>,
    decode_fp8_enabled: bool,
    fused_rms_fp8_enabled: bool,
    maximum_fp8_batch: usize,
}

fn install_linear_fp8(
    runtime: &CudaRuntime,
    weight: &mut LinearWeight,
    site: Option<&Fp8SitePolicy>,
) -> Result<usize> {
    weight.fp8 = None;
    let Some(site) = site.filter(|site| site.enabled) else {
        return Ok(0);
    };
    ensure!(weight.bf16.rank() == 2, "FP8 weight must be rank 2");
    let n = weight.bf16.dims()[0];
    let k = weight.bf16.dims()[1];
    runtime
        .blaslt()
        .prepare_linear_fp8(1, n, k, Fp8ScaleMode::Tensorwide)?;
    weight.fp8 = Some(Fp8LinearWeight {
        data: ops::quantize_weight_e4m3(runtime, &weight.bf16, site.weight_scale)?,
        activation_scale: site.activation_scale,
        weight_scale: site.weight_scale,
    });
    Ok(1)
}

fn linear_dispatch(
    runtime: &CudaRuntime,
    input: &Tensor<bf16>,
    weight: &LinearWeight,
    use_fp8: bool,
) -> Result<Tensor<bf16>> {
    match (use_fp8, weight.fp8.as_ref()) {
        (true, Some(fp8)) => ops::linear_fp8_e4m3(
            runtime,
            input,
            &fp8.data,
            fp8.activation_scale,
            fp8.weight_scale,
        ),
        _ => ops::linear_bf16(runtime, input, &weight.bf16),
    }
}

fn linear_dispatch_input(
    runtime: &CudaRuntime,
    input_bf16: Option<&Tensor<bf16>>,
    input_fp8: Option<&Tensor<u8>>,
    weight: &LinearWeight,
    use_fp8: bool,
) -> Result<Tensor<bf16>> {
    match (use_fp8, weight.fp8.as_ref(), input_fp8) {
        (true, Some(fp8), Some(quantized)) => ops::linear_fp8_e4m3_from_fp8(
            runtime,
            quantized,
            &fp8.data,
            fp8.activation_scale,
            fp8.weight_scale,
        ),
        (true, Some(fp8), None) => {
            let input = input_bf16
                .context("linear_dispatch requires BF16 input when FP8 input is absent")?;
            ops::linear_fp8_e4m3(
                runtime,
                input,
                &fp8.data,
                fp8.activation_scale,
                fp8.weight_scale,
            )
        }
        _ => {
            let input = input_bf16.context("linear_dispatch requires BF16 input for non-FP8 GEMM")?;
            ops::linear_bf16(runtime, input, &weight.bf16)
        }
    }
}

impl Lfm2Model {
    pub(crate) fn resident_weight_bytes(&self) -> usize {
        fn tensor_bf16_bytes(tensor: &Tensor<bf16>) -> usize {
            tensor.numel().saturating_mul(std::mem::size_of::<bf16>())
        }
        fn linear_bytes(weight: &LinearWeight) -> usize {
            tensor_bf16_bytes(&weight.bf16).saturating_add(weight.fp8.as_ref().map_or(0, |fp8| {
                fp8.data.numel().saturating_mul(std::mem::size_of::<u8>())
            }))
        }
        let mut bytes = tensor_bf16_bytes(&self.weights.embedding)
            .saturating_add(tensor_bf16_bytes(&self.weights.final_norm))
            .saturating_add(
                self.inv_freq
                    .numel()
                    .saturating_mul(std::mem::size_of::<f32>()),
            )
            .saturating_add(
                self.weights
                    .lm_head_fp8
                    .as_ref()
                    .map_or(0, |fp8| fp8.data.numel()),
            );
        for layer in &self.weights.layers {
            bytes = bytes
                .saturating_add(tensor_bf16_bytes(&layer.operator_norm))
                .saturating_add(tensor_bf16_bytes(&layer.ffn_norm))
                .saturating_add(linear_bytes(&layer.feed_forward.gate_up))
                .saturating_add(linear_bytes(&layer.feed_forward.down));
            bytes = match &layer.operator {
                OperatorWeights::Conv(conv) => bytes
                    .saturating_add(linear_bytes(&conv.input))
                    .saturating_add(tensor_bf16_bytes(&conv.convolution))
                    .saturating_add(linear_bytes(&conv.output)),
                OperatorWeights::Attention(attention) => bytes
                    .saturating_add(linear_bytes(&attention.query))
                    .saturating_add(linear_bytes(&attention.key))
                    .saturating_add(linear_bytes(&attention.value))
                    .saturating_add(linear_bytes(&attention.output))
                    .saturating_add(tensor_bf16_bytes(&attention.query_norm))
                    .saturating_add(tensor_bf16_bytes(&attention.key_norm)),
            };
        }
        bytes
    }

    pub fn load(runtime: &CudaRuntime, model_dir: &Path) -> Result<Self> {
        let config = Lfm2Config::from_model_dir(model_dir)?;
        let head_dim = config.head_dim();
        let inv_freq_host: Vec<f32> = (0..head_dim / 2)
            .map(|index| {
                config
                    .rope_theta
                    .powf(-2.0 * index as f32 / head_dim as f32)
            })
            .collect();
        let inv_freq = runtime.upload(&inv_freq_host, Shape::new([head_dim / 2]))?;
        for (output, input) in [
            (6144, 2048),
            (2048, 2048),
            (8192, 2048),
            (16384, 2048),
            (2048, 8192),
            (512, 2048),
            (65536, 2048),
        ] {
            runtime.blaslt().prepare_linear_bf16(1, output, input)?;
        }
        let mut weight_store = WeightStore::load(runtime, model_dir)?;
        ensure!(
            weight_store.len() == 148,
            "expected 148 tensors, loaded {}",
            weight_store.len()
        );
        let weights = Lfm2Weights::from_store(runtime, &config, &mut weight_store)?;
        ensure!(
            weights.embedding.dims() == [config.vocab_size, config.hidden_size],
            "embedding weight does not match model config"
        );
        Ok(Self {
            config,
            weights,
            inv_freq,
            decode_fp8_enabled: false,
            fused_rms_fp8_enabled: std::env::var("LFM25_FUSED_RMS_FP8")
                .map(|v| v != "0" && v != "false")
                .unwrap_or(false),
            maximum_fp8_batch: 1,
        })
    }

    pub fn config(&self) -> &Lfm2Config {
        &self.config
    }

    pub(crate) fn decode_fp8_enabled(&self) -> bool {
        self.decode_fp8_enabled
    }

    #[allow(dead_code)]
    pub(crate) fn fused_rms_fp8_enabled(&self) -> bool {
        self.fused_rms_fp8_enabled
    }

    pub(crate) fn set_fused_rms_fp8_enabled(&mut self, enabled: bool) {
        self.fused_rms_fp8_enabled = enabled;
    }

    pub(crate) fn maximum_fp8_batch(&self) -> usize {
        self.maximum_fp8_batch
    }

    pub(crate) fn install_fp8_policy(
        &mut self,
        runtime: &CudaRuntime,
        policy: &Fp8PrecisionPolicy,
    ) -> Result<usize> {
        ensure!(policy.decode_only, "only decode-only FP8 policies are supported");
        let mut by_site = HashMap::with_capacity(policy.sites.len());
        for site in &policy.sites {
            ensure!(
                by_site.insert(site.site.as_str(), site).is_none(),
                "duplicate FP8 policy site {}",
                site.site
            );
        }
        let mut enabled = 0usize;
        for (layer, weights) in self.weights.layers.iter_mut().enumerate() {
            enabled += install_linear_fp8(
                runtime,
                &mut weights.feed_forward.gate_up,
                by_site
                    .get(format!("layers.{layer}.mlp.gate_up").as_str())
                    .copied(),
            )?;
            enabled += install_linear_fp8(
                runtime,
                &mut weights.feed_forward.down,
                by_site
                    .get(format!("layers.{layer}.mlp.down").as_str())
                    .copied(),
            )?;
            match &mut weights.operator {
                OperatorWeights::Conv(conv) => {
                    enabled += install_linear_fp8(
                        runtime,
                        &mut conv.input,
                        by_site
                            .get(format!("layers.{layer}.conv.input").as_str())
                            .copied(),
                    )?;
                    enabled += install_linear_fp8(
                        runtime,
                        &mut conv.output,
                        by_site
                            .get(format!("layers.{layer}.conv.output").as_str())
                            .copied(),
                    )?;
                }
                OperatorWeights::Attention(attention) => {
                    for (name, weight) in [
                        ("query", &mut attention.query),
                        ("key", &mut attention.key),
                        ("value", &mut attention.value),
                        ("output", &mut attention.output),
                    ] {
                        enabled += install_linear_fp8(
                            runtime,
                            weight,
                            by_site
                                .get(format!("layers.{layer}.attention.{name}").as_str())
                                .copied(),
                        )?;
                    }
                }
            }
        }
        self.weights.lm_head_fp8 = match by_site.get("lm_head").copied() {
            Some(site) if site.enabled => {
                runtime.blaslt().prepare_linear_fp8(
                    1,
                    self.weights.embedding.dims()[0],
                    self.weights.embedding.dims()[1],
                    Fp8ScaleMode::Tensorwide,
                )?;
                enabled = enabled.saturating_add(1);
                Some(Fp8LinearWeight {
                    data: ops::quantize_weight_e4m3(
                        runtime,
                        &self.weights.embedding,
                        site.weight_scale,
                    )?,
                    activation_scale: site.activation_scale,
                    weight_scale: site.weight_scale,
                })
            }
            _ => None,
        };
        self.decode_fp8_enabled = enabled > 0;
        self.maximum_fp8_batch = 1;
        Ok(enabled)
    }

    pub(crate) fn prepare_batched_fp8(
        &mut self,
        runtime: &CudaRuntime,
        maximum_batch: usize,
    ) -> Result<()> {
        ensure!(maximum_batch > 0, "FP8 batch limit must be positive");
        ensure!(
            self.has_installed_fp8_weights(),
            "cannot prepare batched FP8 without installed weights"
        );
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
        self.maximum_fp8_batch = maximum_batch;
        Ok(())
    }

    pub(crate) fn restrict_fp8_batch(&mut self, maximum_batch: usize) -> Result<()> {
        ensure!(maximum_batch > 0, "FP8 batch limit must be positive");
        ensure!(
            maximum_batch <= self.maximum_fp8_batch,
            "FP8 batch limit was not prepared"
        );
        self.maximum_fp8_batch = maximum_batch;
        Ok(())
    }

    pub(crate) fn set_decode_fp8_enabled(&mut self, enabled: bool) -> Result<()> {
        ensure!(
            !enabled || self.has_installed_fp8_weights(),
            "cannot enable FP8 decode before installing a policy"
        );
        self.decode_fp8_enabled = enabled;
        Ok(())
    }

    fn has_installed_fp8_weights(&self) -> bool {
        self.weights.lm_head_fp8.is_some()
            || self.weights.layers.iter().any(|layer| {
                layer.feed_forward.gate_up.fp8.is_some()
                    || layer.feed_forward.down.fp8.is_some()
                    || match &layer.operator {
                        OperatorWeights::Conv(conv) => {
                            conv.input.fp8.is_some() || conv.output.fp8.is_some()
                        }
                        OperatorWeights::Attention(attention) => {
                            attention.query.fp8.is_some()
                                || attention.key.fp8.is_some()
                                || attention.value.fp8.is_some()
                                || attention.output.fp8.is_some()
                        }
                    }
            })
    }

    pub fn new_cache(
        &self,
        runtime: &CudaRuntime,
        capacity: usize,
        page_size: KvPageSize,
    ) -> Result<ModelCache> {
        ModelCache::new(runtime, &self.config, capacity, page_size)
    }

    pub fn new_batch_cache(
        &self,
        runtime: &CudaRuntime,
        request_slots: usize,
        maximum_batch_tokens: usize,
        physical_pages: usize,
        page_size: KvPageSize,
    ) -> Result<BatchModelCache> {
        let mut batches = Vec::with_capacity(request_slots.saturating_mul(12));
        batches.extend(1..=request_slots);
        let mut chunk = 1usize;
        while chunk <= maximum_batch_tokens {
            for decode in 0..=request_slots {
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
            for (output, input) in [
                (6144, 2048),
                (2048, 2048),
                (8192, 2048),
                (16384, 2048),
                (2048, 8192),
                (512, 2048),
                (65536, 2048),
            ] {
                runtime.blaslt().prepare_linear_bf16(batch, output, input)?;
            }
        }
        BatchModelCache::new(
            runtime,
            &self.config,
            request_slots,
            maximum_batch_tokens,
            physical_pages,
            page_size,
        )
    }

    pub fn forward_decode_batch(
        &self,
        runtime: &CudaRuntime,
        cache: &mut BatchModelCache,
        token_ids: &[u32],
        positions: &[u32],
        request_slots: &[u32],
    ) -> Result<Tensor<bf16>> {
        ensure!(!token_ids.is_empty(), "decode batch is empty");
        cache.prepare_decode(runtime, token_ids, positions, request_slots)?;
        self.forward_prepared_batch(runtime, cache)
    }

    pub(crate) fn forward_ragged_batch(
        &self,
        runtime: &CudaRuntime,
        cache: &mut BatchModelCache,
        input: RaggedBatchInput<'_>,
    ) -> Result<Tensor<bf16>> {
        ensure!(!input.token_ids.is_empty(), "ragged batch is empty");
        cache.prepare_ragged(runtime, &input)?;
        self.forward_prepared_batch(runtime, cache)
    }

    fn forward_prepared_batch(
        &self,
        runtime: &CudaRuntime,
        cache: &mut BatchModelCache,
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
                    self.attention_batch(
                        runtime,
                        operator,
                        normalized,
                        arena,
                        metadata,
                        cache.is_contiguous_prefill,
                        cache.is_segmented_prefill,
                        cache.max_segment_tokens,
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
                Some(&ffn_input),
                None,
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

    pub fn forward_logits(
        &self,
        runtime: &CudaRuntime,
        cache: &mut ModelCache,
        token_ids: &[u32],
    ) -> Result<Tensor<bf16>> {
        self.forward_logits_instrumented(runtime, cache, token_ids, None, None, None)
    }

    pub(crate) fn forward_logits_profiled(
        &self,
        runtime: &CudaRuntime,
        cache: &mut ModelCache,
        token_ids: &[u32],
        profile: Option<&mut ModelProfileRecorder>,
    ) -> Result<Tensor<bf16>> {
        self.forward_logits_instrumented(runtime, cache, token_ids, profile, None, None)
    }

    pub(crate) fn forward_logits_calibrated(
        &self,
        runtime: &CudaRuntime,
        cache: &mut ModelCache,
        token_ids: &[u32],
        calibration: &mut CalibrationCollector,
    ) -> Result<Tensor<bf16>> {
        self.forward_logits_instrumented(runtime, cache, token_ids, None, Some(calibration), None)
    }

    pub(crate) fn forward_logits_captured(
        &self,
        runtime: &CudaRuntime,
        cache: &mut ModelCache,
        token_ids: &[u32],
        capture: &mut HiddenCapture,
    ) -> Result<Tensor<bf16>> {
        self.forward_logits_instrumented(runtime, cache, token_ids, None, None, Some(capture))
    }

    pub(crate) fn forward_logits_speculative(
        &self,
        runtime: &CudaRuntime,
        cache: &mut ModelCache,
        token_ids: &[u32],
    ) -> Result<(Tensor<bf16>, SpeculativeCheckpoint)> {
        ensure!(
            !token_ids.is_empty(),
            "model forward requires at least one token"
        );
        let num_tokens = token_ids.len();
        let start = cache.sequence_length;
        let next_sequence_length = start
            .checked_add(num_tokens)
            .context("model sequence length overflow")?;
        ensure!(
            next_sequence_length <= cache.capacity,
            "request needs {} cache slots but capacity is {}",
            next_sequence_length,
            cache.capacity
        );

        let use_fp8_decode =
            self.decode_fp8_enabled && start > 0 && num_tokens <= self.maximum_fp8_batch;
        let positions_host: Vec<u32> = (start..next_sequence_length)
            .map(|position| u32::try_from(position).context("position exceeds u32"))
            .collect::<Result<_>>()?;
        let slots_host: Vec<i64> = (start..next_sequence_length)
            .map(|slot| i64::try_from(slot).context("slot exceeds i64"))
            .collect::<Result<_>>()?;
        let tokens = runtime.upload(token_ids, Shape::new([num_tokens]))?;
        let positions = runtime.upload(&positions_host, Shape::new([num_tokens]))?;
        let slots = runtime.upload(&slots_host, Shape::new([num_tokens]))?;

        let mut hidden = ops::embedding_bf16(runtime, &tokens, &self.weights.embedding)?;
        let mut normalized = ops::rms_norm_bf16(
            runtime,
            &hidden,
            &self.weights.layers[0].operator_norm,
            self.config.norm_eps,
        )?;

        let mut conv_histories = Vec::new();
        let mut final_normalized_fp8 = None;

        for layer in 0..self.config.num_hidden_layers {
            let weights = &self.weights.layers[layer];
            let operator_output = match (&weights.operator, &mut cache.layers[layer]) {
                (OperatorWeights::Attention(operator), LayerCache::Attention(kv_cache)) => {
                    let (mut query, key, value) = (
                        linear_dispatch(runtime, &normalized, &operator.query, use_fp8_decode)?
                            .reshape(Shape::new([num_tokens, 32, 64]))?,
                        linear_dispatch(runtime, &normalized, &operator.key, use_fp8_decode)?
                            .reshape(Shape::new([num_tokens, 8, 64]))?,
                        linear_dispatch(runtime, &normalized, &operator.value, use_fp8_decode)?
                            .reshape(Shape::new([num_tokens, 8, 64]))?,
                    );

                    ops::qk_norm_rope_kv_write_decode_bf16(
                        runtime,
                        ops::QkPostprocessInput {
                            query: &mut query,
                            key: &key,
                            value: &value,
                            query_norm: &operator.query_norm,
                            key_norm: &operator.key_norm,
                            inv_freq: &self.inv_freq,
                            position_ids: &positions,
                            slot_mapping: &slots,
                            eps: self.config.norm_eps,
                        },
                        kv_cache,
                    )?;

                    let attended = ops::paged_attention_fast_lfm2_bf16(
                        runtime,
                        &query,
                        kv_cache,
                        &positions,
                    )?
                    .reshape(Shape::new([num_tokens, self.config.hidden_size]))?;
                    linear_dispatch(runtime, &attended, &operator.output, use_fp8_decode)?
                }
                (OperatorWeights::Conv(operator), LayerCache::Conv(state)) => {
                    let projected = linear_dispatch(
                        runtime,
                        &normalized,
                        &operator.input,
                        use_fp8_decode,
                    )?;
                    let mut history = runtime.alloc_bf16(Shape::new([
                        num_tokens,
                        self.config.hidden_size,
                        self.config.conv_l_cache - 1,
                    ]))?;
                    let mixed = ops::short_conv_lfm2_bf16_with_history(
                        runtime,
                        &projected,
                        &operator.convolution,
                        state,
                        &mut history,
                    )?;
                    conv_histories.push((layer, history));
                    linear_dispatch(runtime, &mixed, &operator.output, use_fp8_decode)?
                }
                _ => anyhow::bail!("model/cache layer type mismatch at layer {layer}"),
            };

            let use_fused_rms_fp8_ffn = self.fused_rms_fp8_enabled
                && use_fp8_decode
                && weights.feed_forward.gate_up.fp8.is_some();

            let (post_operator, ffn_input_bf16, ffn_input_fp8) = if use_fused_rms_fp8_ffn {
                let gate_up_fp8 = weights.feed_forward.gate_up.fp8.as_ref().unwrap();
                let mut post_operator = runtime.alloc_bf16(hidden.shape().clone())?;
                let mut ffn_input_fp8 = runtime.alloc_fp8(hidden.shape().clone())?;
                ops::residual_rms_norm_bf16_to_e4m3_into(
                    runtime,
                    &hidden,
                    &operator_output,
                    &weights.ffn_norm,
                    self.config.norm_eps,
                    gate_up_fp8.activation_scale.quantize_multiplier,
                    &mut post_operator,
                    &mut ffn_input_fp8,
                )?;
                (post_operator, None, Some(ffn_input_fp8))
            } else {
                let (post_operator, ffn_input) = ops::residual_rms_norm_bf16(
                    runtime,
                    &hidden,
                    &operator_output,
                    &weights.ffn_norm,
                    self.config.norm_eps,
                )?;
                (post_operator, Some(ffn_input), None)
            };

            let ffn_output = self.feed_forward(
                runtime,
                weights,
                ffn_input_bf16.as_ref(),
                ffn_input_fp8.as_ref(),
                LayerExecution {
                    profile: None,
                    calibration: None,
                    layer,
                    use_fp8: use_fp8_decode,
                },
            )?;

            let is_last_layer = layer + 1 == self.config.num_hidden_layers;
            let next_norm = if !is_last_layer {
                &self.weights.layers[layer + 1].operator_norm
            } else {
                &self.weights.final_norm
            };

            let use_fused_rms_fp8_final = is_last_layer
                && self.fused_rms_fp8_enabled
                && use_fp8_decode
                && self.weights.lm_head_fp8.is_some();

            if use_fused_rms_fp8_final {
                let lm_fp8 = self.weights.lm_head_fp8.as_ref().unwrap();
                let mut next_hidden = runtime.alloc_bf16(post_operator.shape().clone())?;
                let mut final_fp8 = runtime.alloc_fp8(post_operator.shape().clone())?;
                ops::residual_rms_norm_bf16_to_e4m3_into(
                    runtime,
                    &post_operator,
                    &ffn_output,
                    next_norm,
                    self.config.norm_eps,
                    lm_fp8.activation_scale.quantize_multiplier,
                    &mut next_hidden,
                    &mut final_fp8,
                )?;
                hidden = next_hidden;
                final_normalized_fp8 = Some(final_fp8);
            } else {
                (hidden, normalized) = ops::residual_rms_norm_bf16(
                    runtime,
                    &post_operator,
                    &ffn_output,
                    next_norm,
                    self.config.norm_eps,
                )?;
            }
        }

        let logits = match (
            use_fp8_decode,
            self.weights.lm_head_fp8.as_ref(),
            final_normalized_fp8.as_ref(),
        ) {
            (true, Some(fp8), Some(final_fp8)) => ops::linear_fp8_e4m3_from_fp8(
                runtime,
                final_fp8,
                &fp8.data,
                fp8.activation_scale,
                fp8.weight_scale,
            ),
            (true, Some(fp8), None) => ops::linear_fp8_e4m3(
                runtime,
                &normalized,
                &fp8.data,
                fp8.activation_scale,
                fp8.weight_scale,
            ),
            _ => ops::linear_bf16(runtime, &normalized, &self.weights.embedding),
        }?;

        cache.sequence_length = next_sequence_length;
        let checkpoint = SpeculativeCheckpoint {
            start_sequence_length: start,
            num_tokens,
            conv_histories,
        };
        Ok((logits, checkpoint))
    }

    pub(crate) fn rollback_speculative(
        &self,
        runtime: &CudaRuntime,
        cache: &mut ModelCache,
        checkpoint: SpeculativeCheckpoint,
        num_draft_accepted: usize,
    ) -> Result<()> {
        let valid_tokens = 1 + num_draft_accepted;
        ensure!(
            valid_tokens <= checkpoint.num_tokens,
            "cannot accept more tokens than evaluated: valid_tokens={}, num_tokens={}",
            valid_tokens,
            checkpoint.num_tokens
        );
        cache.sequence_length = checkpoint
            .start_sequence_length
            .checked_add(valid_tokens)
            .context("sequence length overflow during speculative rollback")?;

        if valid_tokens < checkpoint.num_tokens {
            let token_idx = valid_tokens - 1;
            for (layer, history) in checkpoint.conv_histories {
                if let LayerCache::Conv(state) = &mut cache.layers[layer] {
                    let elements = state.numel();
                    let source_start = token_idx * elements;
                    runtime.copy_bf16_range(&history, source_start, state, 0, elements)?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn collect_calibration_weights(
        &self,
        runtime: &CudaRuntime,
        calibration: &mut CalibrationCollector,
    ) -> Result<()> {
        calibration.observe(
            runtime,
            "lm_head.weight",
            CalibrationTensorKind::Weight,
            &self.weights.embedding,
        )?;

        for (layer, weights) in self.weights.layers.iter().enumerate() {
            calibration.observe(
                runtime,
                format!("layers.{layer}.mlp.gate_up.weight"),
                CalibrationTensorKind::Weight,
                &weights.feed_forward.gate_up.bf16,
            )?;
            calibration.observe(
                runtime,
                format!("layers.{layer}.mlp.down.weight"),
                CalibrationTensorKind::Weight,
                &weights.feed_forward.down.bf16,
            )?;

            match &weights.operator {
                OperatorWeights::Attention(attention) => {
                    for (name, weight) in [
                        ("query", &attention.query.bf16),
                        ("key", &attention.key.bf16),
                        ("value", &attention.value.bf16),
                        ("output", &attention.output.bf16),
                    ] {
                        calibration.observe(
                            runtime,
                            format!("layers.{layer}.attention.{name}.weight"),
                            CalibrationTensorKind::Weight,
                            weight,
                        )?;
                    }
                }
                OperatorWeights::Conv(conv) => {
                    calibration.observe(
                        runtime,
                        format!("layers.{layer}.conv.input.weight"),
                        CalibrationTensorKind::Weight,
                        &conv.input.bf16,
                    )?;
                    calibration.observe(
                        runtime,
                        format!("layers.{layer}.conv.output.weight"),
                        CalibrationTensorKind::Weight,
                        &conv.output.bf16,
                    )?;
                }
            }
        }

        Ok(())
    }

    pub(crate) fn characterize_calibration_gemms(
        &self,
        runtime: &CudaRuntime,
        calibration: &CalibrationCollector,
    ) -> Result<Fp8GemmErrorReport> {
        let mut reports = Vec::with_capacity(77);

        for (layer, weights) in self.weights.layers.iter().enumerate() {
            for (operation, activation, weight_name, weight) in [
                (
                    "mlp.gate_up",
                    format!("layers.{layer}.mlp.gate_up.input"),
                    format!("layers.{layer}.mlp.gate_up.weight"),
                    &weights.feed_forward.gate_up.bf16,
                ),
                (
                    "mlp.down",
                    format!("layers.{layer}.mlp.down.input"),
                    format!("layers.{layer}.mlp.down.weight"),
                    &weights.feed_forward.down.bf16,
                ),
            ] {
                let site = format!("layers.{layer}.{operation}");
                eprintln!("characterizing real-checkpoint FP8 GEMM {site}");
                reports.push(characterize_gemm_site(
                    runtime,
                    calibration,
                    site,
                    &activation,
                    &weight_name,
                    weight,
                )?);
            }
        }

        eprintln!("characterizing real-checkpoint FP8 GEMM lm_head");
        reports.push(characterize_gemm_site(
            runtime,
            calibration,
            "lm_head".to_string(),
            "lm_head.input",
            "lm_head.weight",
            &self.weights.embedding,
        )?);

        for (layer, weights) in self.weights.layers.iter().enumerate() {
            match &weights.operator {
                OperatorWeights::Conv(conv) => {
                    for (operation, activation, weight_name, weight) in [
                        (
                            "conv.input",
                            format!("layers.{layer}.conv.input.input"),
                            format!("layers.{layer}.conv.input.weight"),
                            &conv.input.bf16,
                        ),
                        (
                            "conv.output",
                            format!("layers.{layer}.conv.output.input"),
                            format!("layers.{layer}.conv.output.weight"),
                            &conv.output.bf16,
                        ),
                    ] {
                        let site = format!("layers.{layer}.{operation}");
                        eprintln!("characterizing real-checkpoint FP8 GEMM {site}");
                        reports.push(characterize_gemm_site(
                            runtime,
                            calibration,
                            site,
                            &activation,
                            &weight_name,
                            weight,
                        )?);
                    }
                }
                OperatorWeights::Attention(attention) => {
                    let activation = format!("layers.{layer}.attention.qkv.input");
                    for (operation, weight_name, weight) in [
                        ("query", "query", &attention.query.bf16),
                        ("key", "key", &attention.key.bf16),
                        ("value", "value", &attention.value.bf16),
                    ] {
                        let site = format!("layers.{layer}.attention.{operation}");
                        let weight_name = format!("layers.{layer}.attention.{weight_name}.weight");
                        eprintln!("characterizing real-checkpoint FP8 GEMM {site}");
                        reports.push(characterize_gemm_site(
                            runtime,
                            calibration,
                            site,
                            &activation,
                            &weight_name,
                            weight,
                        )?);
                    }
                    let site = format!("layers.{layer}.attention.output");
                    let output_activation = format!("layers.{layer}.attention.output.input");
                    let output_weight = format!("layers.{layer}.attention.output.weight");
                    eprintln!("characterizing real-checkpoint FP8 GEMM {site}");
                    reports.push(characterize_gemm_site(
                        runtime,
                        calibration,
                        site,
                        &output_activation,
                        &output_weight,
                        &attention.output.bf16,
                    )?);
                }
            }
        }

        ensure!(
            reports.len() == 77,
            "expected 77 characterized GEMM sites, got {}",
            reports.len()
        );
        Ok(Fp8GemmErrorReport::new(reports))
    }

    fn forward_logits_instrumented(
        &self,
        runtime: &CudaRuntime,
        cache: &mut ModelCache,
        token_ids: &[u32],
        mut profile: Option<&mut ModelProfileRecorder>,
        mut calibration: Option<&mut CalibrationCollector>,
        mut capture: Option<&mut HiddenCapture>,
    ) -> Result<Tensor<bf16>> {
        ensure!(
            !token_ids.is_empty(),
            "model forward requires at least one token"
        );
        let next_sequence_length = cache
            .sequence_length
            .checked_add(token_ids.len())
            .context("model sequence length overflow")?;
        ensure!(
            next_sequence_length <= cache.capacity,
            "request needs {} cache slots but capacity is {}",
            next_sequence_length,
            cache.capacity
        );

        let num_tokens = token_ids.len();
        let start = cache.sequence_length;
        let contiguous_prefill = start == 0 && num_tokens > 1;
        let use_fp8_decode = self.decode_fp8_enabled && start > 0 && num_tokens == 1;
        let positions_host: Vec<u32> = (start..next_sequence_length)
            .map(|position| u32::try_from(position).context("position exceeds u32"))
            .collect::<Result<_>>()?;
        let slots_host: Vec<i64> = (start..next_sequence_length)
            .map(|slot| i64::try_from(slot).context("slot exceeds i64"))
            .collect::<Result<_>>()?;
        let tokens = runtime.upload(token_ids, Shape::new([num_tokens]))?;
        let positions = runtime.upload(&positions_host, Shape::new([num_tokens]))?;
        let slots = runtime.upload(&slots_host, Shape::new([num_tokens]))?;

        let mut hidden = ops::embedding_bf16(runtime, &tokens, &self.weights.embedding)?;
        let mut normalized = profiled(
            runtime,
            profile.as_deref_mut(),
            ProfileRegion::ResidualNorm,
            || {
                ops::rms_norm_bf16(
                    runtime,
                    &hidden,
                    &self.weights.layers[0].operator_norm,
                    self.config.norm_eps,
                )
            },
        )?;

        let mut final_normalized_fp8 = None;

        for layer in 0..self.config.num_hidden_layers {
            let weights = &self.weights.layers[layer];

            if let Some(calibration) = calibration.as_deref_mut() {
                let name = match &weights.operator {
                    OperatorWeights::Attention(_) => {
                        format!("layers.{layer}.attention.qkv.input")
                    }
                    OperatorWeights::Conv(_) => format!("layers.{layer}.conv.input.input"),
                };
                calibration.observe(
                    runtime,
                    name,
                    CalibrationTensorKind::Activation,
                    &normalized,
                )?;
            }

            let coarse = profile
                .as_deref()
                .is_some_and(|profile| profile.mode() == DecodeProfileMode::Coarse);
            let operator_output = match (&weights.operator, &mut cache.layers[layer]) {
                (OperatorWeights::Attention(operator), LayerCache::Attention(kv_cache)) => {
                    let step = AttentionStep {
                        cache: kv_cache,
                        slots: &slots,
                        positions: &positions,
                        contiguous_prefill,
                        context_tokens: next_sequence_length,
                    };
                    if coarse {
                        profiled(
                            runtime,
                            profile.as_deref_mut(),
                            ProfileRegion::Attention,
                            || {
                                self.attention(
                                    runtime,
                                    operator,
                                    &normalized,
                                    step,
                                    LayerExecution {
                                        profile: None,
                                        calibration: calibration.as_deref_mut(),
                                        layer,
                                        use_fp8: use_fp8_decode,
                                    },
                                )
                            },
                        )?
                    } else {
                        self.attention(
                            runtime,
                            operator,
                            &normalized,
                            step,
                            LayerExecution {
                                profile: profile.as_deref_mut(),
                                calibration: calibration.as_deref_mut(),
                                layer,
                                use_fp8: use_fp8_decode,
                            },
                        )?
                    }
                }
                (OperatorWeights::Conv(operator), LayerCache::Conv(state)) => {
                    if coarse {
                        profiled(runtime, profile.as_deref_mut(), ProfileRegion::Conv, || {
                            self.short_conv(
                                runtime,
                                operator,
                                &normalized,
                                state,
                                LayerExecution {
                                    profile: None,
                                    calibration: calibration.as_deref_mut(),
                                    layer,
                                    use_fp8: use_fp8_decode,
                                },
                            )
                        })?
                    } else {
                        self.short_conv(
                            runtime,
                            operator,
                            &normalized,
                            state,
                            LayerExecution {
                                profile: profile.as_deref_mut(),
                                calibration: calibration.as_deref_mut(),
                                layer,
                                use_fp8: use_fp8_decode,
                            },
                        )?
                    }
                }
                _ => anyhow::bail!("model/cache layer type mismatch at layer {layer}"),
            };

            let use_fused_rms_fp8_ffn = self.fused_rms_fp8_enabled
                && use_fp8_decode
                && weights.feed_forward.gate_up.fp8.is_some()
                && calibration.is_none()
                && capture.is_none();

            let (post_operator, ffn_input_bf16, ffn_input_fp8) = if use_fused_rms_fp8_ffn {
                let gate_up_fp8 = weights.feed_forward.gate_up.fp8.as_ref().unwrap();
                let mut post_operator = runtime.alloc_bf16(hidden.shape().clone())?;
                let mut ffn_input_fp8 = runtime.alloc_fp8(hidden.shape().clone())?;
                profiled(
                    runtime,
                    profile.as_deref_mut(),
                    ProfileRegion::ResidualNorm,
                    || {
                        ops::residual_rms_norm_bf16_to_e4m3_into(
                            runtime,
                            &hidden,
                            &operator_output,
                            &weights.ffn_norm,
                            self.config.norm_eps,
                            gate_up_fp8.activation_scale.quantize_multiplier,
                            &mut post_operator,
                            &mut ffn_input_fp8,
                        )
                    },
                )?;
                (post_operator, None, Some(ffn_input_fp8))
            } else {
                let (post_operator, ffn_input) = profiled(
                    runtime,
                    profile.as_deref_mut(),
                    ProfileRegion::ResidualNorm,
                    || {
                        ops::residual_rms_norm_bf16(
                            runtime,
                            &hidden,
                            &operator_output,
                            &weights.ffn_norm,
                            self.config.norm_eps,
                        )
                    },
                )?;
                (post_operator, Some(ffn_input), None)
            };

            if let Some(capture) = capture.as_deref_mut() {
                capture.observe_last_row(
                    runtime,
                    format!("layers.{layer}.post_mixer_residual"),
                    &post_operator,
                )?;
            }
            let ffn_output = if coarse {
                profiled(runtime, profile.as_deref_mut(), ProfileRegion::Mlp, || {
                    self.feed_forward(
                        runtime,
                        weights,
                        ffn_input_bf16.as_ref(),
                        ffn_input_fp8.as_ref(),
                        LayerExecution {
                            profile: None,
                            calibration: calibration.as_deref_mut(),
                            layer,
                            use_fp8: use_fp8_decode,
                        },
                    )
                })?
            } else {
                self.feed_forward(
                    runtime,
                    weights,
                    ffn_input_bf16.as_ref(),
                    ffn_input_fp8.as_ref(),
                    LayerExecution {
                        profile: profile.as_deref_mut(),
                        calibration: calibration.as_deref_mut(),
                        layer,
                        use_fp8: use_fp8_decode,
                    },
                )?
            };
            let is_last_layer = layer + 1 == self.config.num_hidden_layers;
            let next_norm = if !is_last_layer {
                &self.weights.layers[layer + 1].operator_norm
            } else {
                &self.weights.final_norm
            };

            let use_fused_rms_fp8_final = is_last_layer
                && self.fused_rms_fp8_enabled
                && use_fp8_decode
                && self.weights.lm_head_fp8.is_some()
                && capture.is_none()
                && calibration.is_none();

            if use_fused_rms_fp8_final {
                let lm_fp8 = self.weights.lm_head_fp8.as_ref().unwrap();
                let mut next_hidden = runtime.alloc_bf16(post_operator.shape().clone())?;
                let mut final_fp8 = runtime.alloc_fp8(post_operator.shape().clone())?;
                profiled(
                    runtime,
                    profile.as_deref_mut(),
                    ProfileRegion::ResidualNorm,
                    || {
                        ops::residual_rms_norm_bf16_to_e4m3_into(
                            runtime,
                            &post_operator,
                            &ffn_output,
                            next_norm,
                            self.config.norm_eps,
                            lm_fp8.activation_scale.quantize_multiplier,
                            &mut next_hidden,
                            &mut final_fp8,
                        )
                    },
                )?;
                hidden = next_hidden;
                final_normalized_fp8 = Some(final_fp8);
            } else {
                (hidden, normalized) = profiled(
                    runtime,
                    profile.as_deref_mut(),
                    ProfileRegion::ResidualNorm,
                    || {
                        ops::residual_rms_norm_bf16(
                            runtime,
                            &post_operator,
                            &ffn_output,
                            next_norm,
                            self.config.norm_eps,
                        )
                    },
                )?;
            }
            if let Some(capture) = capture.as_deref_mut() {
                capture.observe_last_row(
                    runtime,
                    format!("layers.{layer}.post_ffn_residual"),
                    &hidden,
                )?;
            }
        }

        if let Some(capture) = capture {
            capture.observe_last_row(runtime, "final_rms_norm", &normalized)?;
        }

        if let Some(calibration) = calibration {
            calibration.observe_last_row(runtime, "lm_head.input", &normalized)?;
        }
        let logits = profiled(
            runtime,
            profile,
            ProfileRegion::LmHead,
            || match (
                use_fp8_decode,
                self.weights.lm_head_fp8.as_ref(),
                final_normalized_fp8.as_ref(),
            ) {
                (true, Some(fp8), Some(final_fp8)) => ops::linear_fp8_e4m3_from_fp8(
                    runtime,
                    final_fp8,
                    &fp8.data,
                    fp8.activation_scale,
                    fp8.weight_scale,
                ),
                (true, Some(fp8), None) => ops::linear_last_row_fp8_e4m3(
                    runtime,
                    &normalized,
                    &fp8.data,
                    fp8.activation_scale,
                    fp8.weight_scale,
                ),
                _ => ops::linear_last_row_bf16(runtime, &normalized, &self.weights.embedding),
            },
        )?;
        cache.sequence_length = next_sequence_length;
        Ok(logits)
    }

    fn short_conv(
        &self,
        runtime: &CudaRuntime,
        weights: &ConvWeights,
        normalized: &Tensor<bf16>,
        state: &mut Tensor<bf16>,
        execution: LayerExecution<'_>,
    ) -> Result<Tensor<bf16>> {
        let LayerExecution {
            mut profile,
            calibration,
            layer,
            use_fp8,
        } = execution;
        let projected = profiled(
            runtime,
            profile.as_deref_mut(),
            ProfileRegion::ConvInProj,
            || linear_dispatch(runtime, normalized, &weights.input, use_fp8),
        )?;
        let mixed = profiled(
            runtime,
            profile.as_deref_mut(),
            ProfileRegion::ConvKernel,
            || ops::short_conv_lfm2_bf16(runtime, &projected, &weights.convolution, state),
        )?;
        if let Some(calibration) = calibration {
            calibration.observe(
                runtime,
                format!("layers.{layer}.conv.output.input"),
                CalibrationTensorKind::Activation,
                &mixed,
            )?;
        }
        profiled(
            runtime,
            profile,
            ProfileRegion::ConvOutProj,
            || linear_dispatch(runtime, &mixed, &weights.output, use_fp8),
        )
    }

    fn short_conv_batch(
        &self,
        runtime: &CudaRuntime,
        weights: &ConvWeights,
        normalized: &Tensor<bf16>,
        states: &mut Tensor<bf16>,
        metadata: &GpuBatch,
        use_fp8: bool,
    ) -> Result<Tensor<bf16>> {
        let projected = linear_dispatch(runtime, normalized, &weights.input, use_fp8)?;
        let mixed = ops::short_conv_segmented_lfm2_bf16(
            runtime,
            &projected,
            &weights.convolution,
            states,
            metadata.segment_offsets(),
            metadata.segment_slots(),
        )?;
        linear_dispatch(runtime, &mixed, &weights.output, use_fp8)
    }

    fn feed_forward(
        &self,
        runtime: &CudaRuntime,
        weights: &LayerWeights,
        input_bf16: Option<&Tensor<bf16>>,
        input_fp8: Option<&Tensor<u8>>,
        execution: LayerExecution<'_>,
    ) -> Result<Tensor<bf16>> {
        let LayerExecution {
            mut profile,
            mut calibration,
            layer,
            use_fp8,
        } = execution;
        if let Some(calibration) = calibration.as_deref_mut() {
            if let Some(input) = input_bf16 {
                calibration.observe(
                    runtime,
                    format!("layers.{layer}.mlp.gate_up.input"),
                    CalibrationTensorKind::Activation,
                    input,
                )?;
            }
        }
        let gate_up = profiled(
            runtime,
            profile.as_deref_mut(),
            ProfileRegion::MlpGateUpGemm,
            || {
                linear_dispatch_input(
                    runtime,
                    input_bf16,
                    input_fp8,
                    &weights.feed_forward.gate_up,
                    use_fp8,
                )
            },
        )?;
        let activated = profiled(
            runtime,
            profile.as_deref_mut(),
            ProfileRegion::MlpSilu,
            || ops::silu_mul_packed_bf16(runtime, &gate_up),
        )?;
        if let Some(calibration) = calibration {
            calibration.observe(
                runtime,
                format!("layers.{layer}.mlp.down.input"),
                CalibrationTensorKind::Activation,
                &activated,
            )?;
        }
        profiled(
            runtime,
            profile,
            ProfileRegion::MlpDownGemm,
            || linear_dispatch(runtime, &activated, &weights.feed_forward.down, use_fp8),
        )
    }

    fn attention(
        &self,
        runtime: &CudaRuntime,
        weights: &AttentionWeights,
        normalized: &Tensor<bf16>,
        step: AttentionStep<'_>,
        execution: LayerExecution<'_>,
    ) -> Result<Tensor<bf16>> {
        let AttentionStep {
            cache,
            slots,
            positions,
            contiguous_prefill,
            context_tokens,
        } = step;
        let LayerExecution {
            mut profile,
            calibration,
            layer,
            use_fp8,
        } = execution;
        let num_tokens = normalized.dims()[0];
        let (mut query, mut key, value) = profiled(
            runtime,
            profile.as_deref_mut(),
            ProfileRegion::AttnQkvProj,
            || {
                Ok((
                    linear_dispatch(runtime, normalized, &weights.query, use_fp8)?
                        .reshape(Shape::new([num_tokens, 32, 64]))?,
                    linear_dispatch(runtime, normalized, &weights.key, use_fp8)?
                        .reshape(Shape::new([num_tokens, 8, 64]))?,
                    linear_dispatch(runtime, normalized, &weights.value, use_fp8)?
                        .reshape(Shape::new([num_tokens, 8, 64]))?,
                ))
            },
        )?;

        profiled(
            runtime,
            profile.as_deref_mut(),
            ProfileRegion::AttnPostprocess,
            || {
                if contiguous_prefill {
                    query = ops::rms_norm_bf16(
                        runtime,
                        &query,
                        &weights.query_norm,
                        self.config.norm_eps,
                    )?;
                    key = ops::rms_norm_bf16(
                        runtime,
                        &key,
                        &weights.key_norm,
                        self.config.norm_eps,
                    )?;
                    ops::rope_qk_bf16_inplace(
                        runtime,
                        &mut query,
                        &mut key,
                        &self.inv_freq,
                        positions,
                    )?;
                    cache.write_lfm2(runtime, &key, &value, slots)
                } else if ops::should_use_mok_one_kernel(
                    cache.page_size().value(),
                    context_tokens,
                    1,
                ) {
                    Ok(())
                } else {
                    ops::qk_norm_rope_kv_write_decode_bf16(
                        runtime,
                        ops::QkPostprocessInput {
                            query: &mut query,
                            key: &key,
                            value: &value,
                            query_norm: &weights.query_norm,
                            key_norm: &weights.key_norm,
                            inv_freq: &self.inv_freq,
                            position_ids: positions,
                            slot_mapping: slots,
                            eps: self.config.norm_eps,
                        },
                        cache,
                    )
                }
            },
        )?;
        let attended = profiled(
            runtime,
            profile.as_deref_mut(),
            ProfileRegion::AttnXqa,
            || {
                if contiguous_prefill {
                    ops::prefill_attention_lfm2_bf16(runtime, &query, &key, &value)
                } else if ops::should_use_mok_one_kernel(
                    cache.page_size().value(),
                    context_tokens,
                    1,
                ) {
                    ops::fused_paged_attention_decode_lfm2_bf16(
                        runtime,
                        ops::FusedPagedAttentionInput {
                            attention: ops::FusedAttentionInput {
                                query_raw: &query,
                                key_raw: &key,
                                value_raw: &value,
                                query_norm: &weights.query_norm,
                                key_norm: &weights.key_norm,
                                inv_freq: &self.inv_freq,
                                position_ids: positions,
                                slot_mapping: slots,
                                eps: self.config.norm_eps,
                            },
                            cache,
                        },
                    )
                } else {
                    ops::paged_attention_fast_lfm2_bf16(runtime, &query, cache, positions)
                }
            },
        )?
        .reshape(Shape::new([num_tokens, self.config.hidden_size]))?;
        if let Some(calibration) = calibration {
            calibration.observe(
                runtime,
                format!("layers.{layer}.attention.output.input"),
                CalibrationTensorKind::Activation,
                &attended,
            )?;
        }
        profiled(
            runtime,
            profile,
            ProfileRegion::AttnOutProj,
            || linear_dispatch(runtime, &attended, &weights.output, use_fp8),
        )
    }

    fn attention_batch(
        &self,
        runtime: &CudaRuntime,
        weights: &AttentionWeights,
        normalized: Tensor<bf16>,
        arena: &mut PagedKvArena,
        metadata: &GpuBatch,
        is_contiguous_prefill: bool,
        is_segmented_prefill: bool,
        max_segment_tokens: usize,
        use_fp8: bool,
    ) -> Result<Tensor<bf16>> {
        let num_tokens = normalized.dims()[0];
        let mut query = linear_dispatch(runtime, &normalized, &weights.query, use_fp8)?
            .reshape(Shape::new([num_tokens, 32, 64]))?;
        let mut key = linear_dispatch(runtime, &normalized, &weights.key, use_fp8)?
            .reshape(Shape::new([num_tokens, 8, 64]))?;
        let value = linear_dispatch(runtime, &normalized, &weights.value, use_fp8)?
            .reshape(Shape::new([num_tokens, 8, 64]))?;

        let decode_only = metadata.segment_slots().numel() == num_tokens
            && metadata.segment_offsets().numel() == num_tokens + 1;
        let one_kernel_decode = decode_only
            && ops::should_use_mok_one_kernel(
                arena.page_size().value(),
                metadata.max_context_tokens(),
                num_tokens,
            );
        let attended = if one_kernel_decode {
            ops::fused_ragged_paged_attention_decode_lfm2_bf16(
                runtime,
                ops::FusedRaggedAttentionInput {
                    attention: ops::FusedAttentionInput {
                        query_raw: &query,
                        key_raw: &key,
                        value_raw: &value,
                        query_norm: &weights.query_norm,
                        key_norm: &weights.key_norm,
                        inv_freq: &self.inv_freq,
                        position_ids: metadata.positions(),
                        slot_mapping: metadata.physical_slots(),
                        eps: self.config.norm_eps,
                    },
                    arena,
                    block_tables: metadata.block_tables(),
                    block_table_stride: metadata.block_table_stride(),
                    request_slots: metadata.request_slots(),
                },
            )?
        } else if decode_only {
            ops::qk_norm_rope_kv_write_arena_decode_bf16(
                runtime,
                ops::QkPostprocessInput {
                    query: &mut query,
                    key: &key,
                    value: &value,
                    query_norm: &weights.query_norm,
                    key_norm: &weights.key_norm,
                    inv_freq: &self.inv_freq,
                    position_ids: metadata.positions(),
                    slot_mapping: metadata.physical_slots(),
                    eps: self.config.norm_eps,
                },
                arena,
            )?;
            ops::paged_ragged_attention_fast_lfm2_bf16(
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
        } else {
            query = ops::rms_norm_bf16(
                runtime,
                &query,
                &weights.query_norm,
                self.config.norm_eps,
            )?;
            key = ops::rms_norm_bf16(
                runtime,
                &key,
                &weights.key_norm,
                self.config.norm_eps,
            )?;
            ops::rope_qk_bf16_inplace(
                runtime,
                &mut query,
                &mut key,
                &self.inv_freq,
                metadata.positions(),
            )?;
            arena.write_lfm2(runtime, &key, &value, metadata.physical_slots())?;
            if is_contiguous_prefill {
                ops::prefill_attention_lfm2_bf16(runtime, &query, &key, &value)?
            } else if is_segmented_prefill && ops::prefill_dispatch::flash_prefill_enabled() {
                ops::segmented_prefill_attention_lfm2_bf16(
                    runtime,
                    &query,
                    &key,
                    &value,
                    metadata.segment_offsets(),
                    metadata.segment_offsets().numel() - 1,
                    max_segment_tokens,
                )?
            } else {
                ops::hybrid_ragged_attention_lfm2_bf16(
                    runtime,
                    ops::HybridRaggedAttentionInput {
                        query: &query,
                        current_key: &key,
                        current_value: &value,
                        arena,
                        block_tables: metadata.block_tables(),
                        block_table_stride: metadata.block_table_stride(),
                        request_slots: metadata.request_slots(),
                        position_ids: metadata.positions(),
                        segment_offsets: metadata.segment_offsets(),
                    },
                )?
            }
        }
        .reshape(Shape::new([num_tokens, self.config.hidden_size]))?;
        linear_dispatch(runtime, &attended, &weights.output, use_fp8)
    }
}
