use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cache::PagedKvCache,
    cuda::CudaRuntime,
    tensor::Tensor,
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn qk_norm_rope_kv_write_decode_bf16(
    runtime: &CudaRuntime,
    query: &mut Tensor<bf16>,
    key: &Tensor<bf16>,
    value: &Tensor<bf16>,
    query_norm: &Tensor<bf16>,
    key_norm: &Tensor<bf16>,
    inv_freq: &Tensor<f32>,
    position_ids: &Tensor<u32>,
    slot_mapping: &Tensor<i64>,
    cache: &mut PagedKvCache,
    eps: f32,
) -> Result<()> {
    ensure!(
        query.rank() == 3 && query.dims()[1..] == [32, 64],
        "fused query must have shape [N,32,64]"
    );
    let num_tokens = query.dims()[0];
    ensure!(
        key.dims() == [num_tokens, 8, 64],
        "fused key must have shape [N,8,64]"
    );
    ensure!(value.shape() == key.shape(), "fused K/V shape mismatch");
    ensure!(query_norm.numel() == 64, "query norm weight must have 64 elements");
    ensure!(key_norm.numel() == 64, "key norm weight must have 64 elements");
    ensure!(inv_freq.numel() == 32, "RoPE inv_freq must have 32 elements");
    ensure!(position_ids.numel() == num_tokens, "position count mismatch");
    ensure!(slot_mapping.numel() == num_tokens, "slot mapping count mismatch");

    let page_size = cache.page_size().value();
    let num_pages = cache.num_pages();
    let (key_cache, value_cache) = cache.kv_mut();

    unsafe {
        runtime.kernels().qk_postprocess().launch_decode(
            runtime.stream(),
            page_size,
            query.storage_mut(),
            key.storage(),
            value.storage(),
            query_norm.storage(),
            key_norm.storage(),
            inv_freq.storage(),
            position_ids.storage(),
            slot_mapping.storage(),
            key_cache.storage_mut(),
            value_cache.storage_mut(),
            num_tokens,
            num_pages,
            eps,
        )?;
    }

    Ok(())
}
