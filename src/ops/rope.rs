use anyhow::{Context as _, Result, ensure};
use half::bf16;

use crate::{
    cuda::{CudaRuntime, RopeLaunch},
    tensor::Tensor,
};

const MAX_HEAD_DIM: usize = 512;

pub fn rope_qk_bf16_inplace(
    runtime: &CudaRuntime,
    query: &mut Tensor<bf16>,
    key: &mut Tensor<bf16>,
    inv_freq: &Tensor<f32>,
    position_ids: &Tensor<u32>,
) -> Result<()> {
    ensure!(
        query.rank() >= 3,
        "RoPE query must have rank >= 3, got {:?}",
        query.dims()
    );
    ensure!(
        key.rank() >= 3,
        "RoPE key must have rank >= 3, got {:?}",
        key.dims()
    );
    ensure!(query.numel() > 0, "RoPE query must not be empty");
    ensure!(key.numel() > 0, "RoPE key must not be empty");

    let query_rank = query.rank();
    let key_rank = key.rank();
    let query_dims = query.dims();
    let key_dims = key.dims();
    let num_q_heads = query_dims[query_rank - 2];
    let num_kv_heads = key_dims[key_rank - 2];
    let query_head_dim = query_dims[query_rank - 1];
    let key_head_dim = key_dims[key_rank - 1];

    ensure!(num_q_heads > 0, "RoPE num_q_heads must be > 0");
    ensure!(num_kv_heads > 0, "RoPE num_kv_heads must be > 0");
    ensure!(query_head_dim > 0, "RoPE head_dim must be > 0");
    ensure!(
        query_head_dim == key_head_dim,
        "RoPE Q/K head_dim mismatch: query={}, key={}",
        query_head_dim,
        key_head_dim
    );

    let head_dim = query_head_dim;
    ensure!(
        head_dim.is_multiple_of(2),
        "RoPE head_dim must be even, got {head_dim}"
    );
    ensure!(
        head_dim <= MAX_HEAD_DIM,
        "RoPE head_dim={head_dim} exceeds supported maximum {MAX_HEAD_DIM}"
    );

    let query_prefix = &query_dims[..query_rank - 2];
    let key_prefix = &key_dims[..key_rank - 2];
    ensure!(
        query_prefix == key_prefix,
        "RoPE Q/K token dimensions mismatch: query prefix={:?}, key prefix={:?}",
        query_prefix,
        key_prefix
    );

    let q_values_per_token = num_q_heads
        .checked_mul(head_dim)
        .context("validated tensor shape overflow")?;
    ensure!(
        query.numel().is_multiple_of(q_values_per_token),
        "invalid RoPE query layout"
    );
    let num_tokens = query.numel() / q_values_per_token;
    ensure!(num_tokens > 0, "RoPE requires at least one token");

    let k_values_per_token = num_kv_heads
        .checked_mul(head_dim)
        .context("validated tensor shape overflow")?;
    ensure!(
        key.numel() == num_tokens * k_values_per_token,
        "invalid RoPE key layout"
    );
    ensure!(
        inv_freq.rank() == 1,
        "RoPE inv_freq must have rank 1, got {:?}",
        inv_freq.dims()
    );

    let half_dim = head_dim / 2;
    ensure!(
        inv_freq.numel() == half_dim,
        "RoPE inv_freq mismatch: expected {}, got {:?}",
        half_dim,
        inv_freq.dims()
    );
    ensure!(
        position_ids.numel() == num_tokens,
        "RoPE position_ids mismatch: expected {num_tokens} positions, got {:?}",
        position_ids.dims()
    );

    unsafe {
        runtime.kernels().rope().launch_qk_bf16_inplace(
            runtime.stream(),
            RopeLaunch {
                query: query.storage_mut(),
                key: key.storage_mut(),
                inv_freq: inv_freq.storage(),
                position_ids: position_ids.storage(),
                num_tokens,
                num_q_heads,
                num_kv_heads,
                head_dim,
            },
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use half::bf16;

    use crate::{cuda::CudaRuntime, tensor::Shape};

    use super::*;

    fn make_inv_freq(head_dim: usize, theta: f32) -> Vec<f32> {
        (0..head_dim / 2)
            .map(|index| theta.powf(-2.0 * index as f32 / head_dim as f32))
            .collect()
    }

    #[test]
    fn rope_accepts_batched_prefix() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        const BATCH: usize = 2;
        const SEQ: usize = 3;
        const Q_HEADS: usize = 4;
        const KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 8;

        let query_numel = BATCH * SEQ * Q_HEADS * HEAD_DIM;
        let key_numel = BATCH * SEQ * KV_HEADS * HEAD_DIM;
        let mut query = runtime.upload(
            &vec![bf16::from_f32(1.0); query_numel],
            Shape::new([BATCH, SEQ, Q_HEADS, HEAD_DIM]),
        )?;
        let mut key = runtime.upload(
            &vec![bf16::from_f32(1.0); key_numel],
            Shape::new([BATCH, SEQ, KV_HEADS, HEAD_DIM]),
        )?;
        let inv_freq_host = make_inv_freq(HEAD_DIM, 1_000_000.0);
        let inv_freq = runtime.upload(&inv_freq_host, Shape::new([HEAD_DIM / 2]))?;
        let position_host = [0u32, 1, 2, 0, 1, 2];
        let positions = runtime.upload(&position_host, Shape::new([BATCH, SEQ]))?;

        rope_qk_bf16_inplace(&runtime, &mut query, &mut key, &inv_freq, &positions)?;
        assert_eq!(query.dims(), &[BATCH, SEQ, Q_HEADS, HEAD_DIM]);
        assert_eq!(key.dims(), &[BATCH, SEQ, KV_HEADS, HEAD_DIM]);
        Ok(())
    }

    #[test]
    fn rope_rejects_qk_prefix_mismatch() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let mut query = runtime.upload(
            &[bf16::from_f32(1.0); 2 * 3 * 4 * 8],
            Shape::new([2, 3, 4, 8]),
        )?;
        let mut key = runtime.upload(
            &[bf16::from_f32(1.0); 2 * 4 * 2 * 8],
            Shape::new([2, 4, 2, 8]),
        )?;
        let inv_freq = runtime.upload(&[1.0f32; 4], Shape::new([4]))?;
        let positions = runtime.upload(&[0u32; 6], Shape::new([6]))?;
        assert!(
            rope_qk_bf16_inplace(&runtime, &mut query, &mut key, &inv_freq, &positions).is_err()
        );
        Ok(())
    }

    #[test]
    fn rope_rejects_inv_freq_mismatch() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let mut query = runtime.upload(&[bf16::from_f32(1.0); 4 * 8], Shape::new([1, 4, 8]))?;
        let mut key = runtime.upload(&[bf16::from_f32(1.0); 2 * 8], Shape::new([1, 2, 8]))?;
        let inv_freq = runtime.upload(&[1.0f32; 3], Shape::new([3]))?;
        let positions = runtime.upload(&[0u32], Shape::new([1]))?;
        assert!(
            rope_qk_bf16_inplace(&runtime, &mut query, &mut key, &inv_freq, &positions).is_err()
        );
        Ok(())
    }
}
