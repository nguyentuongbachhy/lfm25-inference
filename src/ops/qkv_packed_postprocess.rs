use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cache::PagedKvArena,
    cuda::{CudaRuntime, PackedQkvPostprocessLaunch},
    tensor::{Shape, Tensor},
};

pub(crate) struct PackedQkvPostprocessInput<'a> {
    pub(crate) packed_qkv: &'a Tensor<bf16>,
    pub(crate) query_norm: &'a Tensor<bf16>,
    pub(crate) key_norm: &'a Tensor<bf16>,
    pub(crate) inv_freq: &'a Tensor<f32>,
    pub(crate) position_ids: &'a Tensor<u32>,
    pub(crate) slot_mapping: &'a Tensor<i64>,
    pub(crate) eps: f32,
}

pub(crate) fn qk_norm_rope_kv_write_arena_packed_decode_bf16(
    runtime: &CudaRuntime,
    input: PackedQkvPostprocessInput<'_>,
    arena: &mut PagedKvArena,
) -> Result<Tensor<bf16>> {
    let PackedQkvPostprocessInput {
        packed_qkv,
        query_norm,
        key_norm,
        inv_freq,
        position_ids,
        slot_mapping,
        eps,
    } = input;
    ensure!(
        packed_qkv.rank() == 2 && packed_qkv.dims()[1] == 3072,
        "packed QKV must have shape [N,3072], got {:?}",
        packed_qkv.dims()
    );
    let num_tokens = packed_qkv.dims()[0];
    ensure!(num_tokens > 0, "packed QKV decode requires tokens");
    ensure!(query_norm.numel() == 64, "query norm weight must have 64 elements");
    ensure!(key_norm.numel() == 64, "key norm weight must have 64 elements");
    ensure!(inv_freq.numel() == 32, "RoPE inv_freq must have 32 elements");
    ensure!(position_ids.numel() == num_tokens, "position count mismatch");
    ensure!(slot_mapping.numel() == num_tokens, "slot mapping count mismatch");

    let mut query = runtime.alloc_bf16(Shape::new([num_tokens, 32, 64]))?;
    let page_size = arena.page_size().value();
    let num_pages = arena.num_pages();
    let (key_cache, value_cache) = arena.kv_mut();
    unsafe {
        runtime.kernels().qkv_packed_postprocess().launch_decode(
            runtime.stream(),
            PackedQkvPostprocessLaunch {
                page_size,
                packed_qkv: packed_qkv.storage(),
                query_out: query.storage_mut(),
                query_norm: query_norm.storage(),
                key_norm: key_norm.storage(),
                inv_freq: inv_freq.storage(),
                position_ids: position_ids.storage(),
                slot_mapping: slot_mapping.storage(),
                key_cache: key_cache.storage_mut(),
                value_cache: value_cache.storage_mut(),
                num_tokens,
                num_pages,
                eps,
            },
        )?;
    }
    Ok(query)
}
