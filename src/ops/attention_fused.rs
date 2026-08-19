use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cache::{PagedKvArena, PagedKvCache},
    cuda::{CudaRuntime, FusedAttentionCommon, FusedDecodeLaunch, FusedRaggedDecodeLaunch},
    tensor::{Shape, Tensor},
};

pub(crate) struct FusedAttentionInput<'a> {
    pub(crate) query_raw: &'a Tensor<bf16>,
    pub(crate) key_raw: &'a Tensor<bf16>,
    pub(crate) value_raw: &'a Tensor<bf16>,
    pub(crate) query_norm: &'a Tensor<bf16>,
    pub(crate) key_norm: &'a Tensor<bf16>,
    pub(crate) inv_freq: &'a Tensor<f32>,
    pub(crate) position_ids: &'a Tensor<u32>,
    pub(crate) slot_mapping: &'a Tensor<i64>,
    pub(crate) eps: f32,
}

pub(crate) struct FusedPagedAttentionInput<'a> {
    pub(crate) attention: FusedAttentionInput<'a>,
    pub(crate) cache: &'a mut PagedKvCache,
}

pub(crate) struct FusedRaggedAttentionInput<'a> {
    pub(crate) attention: FusedAttentionInput<'a>,
    pub(crate) arena: &'a mut PagedKvArena,
    pub(crate) block_tables: &'a Tensor<u32>,
    pub(crate) block_table_stride: usize,
    pub(crate) request_slots: &'a Tensor<u32>,
}

pub(crate) fn fused_paged_attention_decode_lfm2_bf16(
    runtime: &CudaRuntime,
    input: FusedPagedAttentionInput<'_>,
) -> Result<Tensor<bf16>> {
    let num_tokens = input.attention.query_raw.dims()[0];
    let mut output = runtime.alloc_bf16(Shape::new([num_tokens, 32, 64]))?;
    fused_paged_attention_decode_lfm2_bf16_into(runtime, input, &mut output)?;
    Ok(output)
}

pub(crate) fn fused_paged_attention_decode_lfm2_bf16_into(
    runtime: &CudaRuntime,
    input: FusedPagedAttentionInput<'_>,
    output: &mut Tensor<bf16>,
) -> Result<()> {
    let FusedPagedAttentionInput { attention, cache } = input;
    let num_tokens = validate_inputs(&attention)?;
    output.set_logical_shape(Shape::new([num_tokens, 32, 64]))?;
    let page_size = cache.page_size().value();
    let num_pages = cache.num_pages();
    let (block_table, key_cache, value_cache) = cache.attention_parts_mut();

    unsafe {
        runtime.kernels().attention_fused().launch_decode(
            runtime.stream(),
            FusedDecodeLaunch {
                common: FusedAttentionCommon {
                    page_size,
                    query_raw: attention.query_raw.storage(),
                    key_raw: attention.key_raw.storage(),
                    value_raw: attention.value_raw.storage(),
                    query_norm: attention.query_norm.storage(),
                    key_norm: attention.key_norm.storage(),
                    inv_freq: attention.inv_freq.storage(),
                    key_cache: key_cache.storage_mut(),
                    value_cache: value_cache.storage_mut(),
                    position_ids: attention.position_ids.storage(),
                    slot_mapping: attention.slot_mapping.storage(),
                    output: output.storage_mut(),
                    num_tokens,
                    num_pages,
                    eps: attention.eps,
                },
                block_table: block_table.storage(),
            },
        )?;
    }
    Ok(())
}

pub(crate) fn fused_ragged_paged_attention_decode_lfm2_bf16(
    runtime: &CudaRuntime,
    input: FusedRaggedAttentionInput<'_>,
) -> Result<Tensor<bf16>> {
    let num_tokens = input.attention.query_raw.dims()[0];
    let mut output = runtime.alloc_bf16(Shape::new([num_tokens, 32, 64]))?;
    fused_ragged_paged_attention_decode_lfm2_bf16_into(runtime, input, &mut output)?;
    Ok(output)
}

pub(crate) fn fused_ragged_paged_attention_decode_lfm2_bf16_into(
    runtime: &CudaRuntime,
    input: FusedRaggedAttentionInput<'_>,
    output: &mut Tensor<bf16>,
) -> Result<()> {
    let FusedRaggedAttentionInput {
        attention,
        arena,
        block_tables,
        block_table_stride,
        request_slots,
    } = input;
    let num_tokens = validate_inputs(&attention)?;
    ensure!(block_tables.rank() == 2, "fused ragged block tables must be rank 2");
    ensure!(
        block_tables.dims()[1] == block_table_stride,
        "fused ragged block table stride/shape mismatch"
    );
    ensure!(request_slots.numel() == num_tokens, "fused ragged request slot count mismatch");
    output.set_logical_shape(Shape::new([num_tokens, 32, 64]))?;

    let page_size = arena.page_size().value();
    let num_pages = arena.num_pages();
    let (key_cache, value_cache) = arena.kv_mut();
    unsafe {
        runtime.kernels().attention_fused().launch_ragged_decode(
            runtime.stream(),
            FusedRaggedDecodeLaunch {
                common: FusedAttentionCommon {
                    page_size,
                    query_raw: attention.query_raw.storage(),
                    key_raw: attention.key_raw.storage(),
                    value_raw: attention.value_raw.storage(),
                    query_norm: attention.query_norm.storage(),
                    key_norm: attention.key_norm.storage(),
                    inv_freq: attention.inv_freq.storage(),
                    key_cache: key_cache.storage_mut(),
                    value_cache: value_cache.storage_mut(),
                    position_ids: attention.position_ids.storage(),
                    slot_mapping: attention.slot_mapping.storage(),
                    output: output.storage_mut(),
                    num_tokens,
                    num_pages,
                    eps: attention.eps,
                },
                block_tables: block_tables.storage(),
                request_slots: request_slots.storage(),
                block_table_stride,
            },
        )?;
    }
    Ok(())
}

fn validate_inputs(input: &FusedAttentionInput<'_>) -> Result<usize> {
    ensure!(
        input.query_raw.rank() == 3 && input.query_raw.dims()[1..] == [32, 64],
        "fused LFM2 query must have shape [N,32,64], got {:?}",
        input.query_raw.dims()
    );
    let num_tokens = input.query_raw.dims()[0];
    ensure!(num_tokens > 0, "fused attention requires at least one token");
    ensure!(
        input.key_raw.dims() == [num_tokens, 8, 64],
        "fused LFM2 key must have shape [{num_tokens},8,64], got {:?}",
        input.key_raw.dims()
    );
    ensure!(input.value_raw.shape() == input.key_raw.shape(), "fused LFM2 K/V shape mismatch");
    ensure!(
        input.query_norm.rank() == 1 && input.query_norm.numel() == 64,
        "fused query norm weight must have shape [64]"
    );
    ensure!(
        input.key_norm.rank() == 1 && input.key_norm.numel() == 64,
        "fused key norm weight must have shape [64]"
    );
    ensure!(
        input.inv_freq.rank() == 1 && input.inv_freq.numel() == 32,
        "fused RoPE inv_freq must have shape [32]"
    );
    ensure!(input.position_ids.numel() == num_tokens, "fused attention position count mismatch");
    ensure!(input.slot_mapping.numel() == num_tokens, "fused attention slot mapping count mismatch");
    ensure!(
        input.eps.is_finite() && input.eps >= 0.0,
        "fused attention epsilon must be finite and non-negative"
    );
    Ok(num_tokens)
}
