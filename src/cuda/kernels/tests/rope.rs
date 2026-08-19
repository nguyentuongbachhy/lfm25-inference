use anyhow::Result;
use half::bf16;

use crate::{
    cuda::{
        CudaRuntime, RopeLaunch,
        benchmark::{BenchConfig, benchmark_gpu},
        testing::{assert_close_bf16, assert_eq_bf16, readback},
    },
    tensor::Shape,
};

fn make_inv_freq(head_dim: usize, theta: f32) -> Vec<f32> {
    let half_dim = head_dim / 2;
    (0..half_dim)
        .map(|index| {
            let exponent = -2.0 * index as f32 / head_dim as f32;
            theta.powf(exponent)
        })
        .collect()
}

fn make_values(n: usize, multiplier: usize, modulus: usize) -> Vec<bf16> {
    (0..n)
        .map(|index| {
            let value =
                ((index.wrapping_mul(multiplier) % modulus) as f32 - modulus as f32 * 0.5) / 32.0;
            bf16::from_f32(value)
        })
        .collect()
}

fn rope_reference_inplace(
    tensor: &mut [bf16],
    inv_freq: &[f32],
    position_ids: &[u32],
    num_heads: usize,
    head_dim: usize,
) {
    let half_dim = head_dim / 2;
    for (token, &position_id) in position_ids.iter().enumerate() {
        let position = position_id as f32;
        for head in 0..num_heads {
            let base = (token * num_heads + head) * head_dim;
            for (pair, &frequency) in inv_freq.iter().take(half_dim).enumerate() {
                let angle = position * frequency;
                let sin_value = angle.sin();
                let cos_value = angle.cos();
                let idx1 = base + pair;
                let idx2 = idx1 + half_dim;
                let x1 = tensor[idx1].to_f32();
                let x2 = tensor[idx2].to_f32();
                let y1 = (-x2).mul_add(sin_value, x1 * cos_value);
                let y2 = x1.mul_add(sin_value, x2 * cos_value);
                tensor[idx1] = bf16::from_f32(y1);
                tensor[idx2] = bf16::from_f32(y2);
            }
        }
    }
}

fn run_case(
    runtime: &CudaRuntime,
    num_tokens: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    let query_numel = num_tokens * num_q_heads * head_dim;
    let key_numel = num_tokens * num_kv_heads * head_dim;
    let query_host = make_values(query_numel, 37, 257);
    let key_host = make_values(key_numel, 17, 193);
    let position_ids: Vec<u32> = (0..num_tokens)
        .map(|token| (token * 7 + 3) as u32)
        .collect();
    let inv_freq = make_inv_freq(head_dim, 1_000_000.0);
    let mut expected_query = query_host.clone();
    let mut expected_key = key_host.clone();

    rope_reference_inplace(
        &mut expected_query,
        &inv_freq,
        &position_ids,
        num_q_heads,
        head_dim,
    );
    rope_reference_inplace(
        &mut expected_key,
        &inv_freq,
        &position_ids,
        num_kv_heads,
        head_dim,
    );

    let mut query = runtime.upload(&query_host, Shape::new([num_tokens, num_q_heads, head_dim]))?;
    let mut key = runtime.upload(&key_host, Shape::new([num_tokens, num_kv_heads, head_dim]))?;
    let inv_freq = runtime.upload(&inv_freq, Shape::new([head_dim / 2]))?;
    let position_ids = runtime.upload(&position_ids, Shape::new([num_tokens]))?;

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

    let actual_query = readback(runtime, &query)?;
    let actual_key = readback(runtime, &key)?;
    assert_close_bf16(&actual_query, &expected_query, 0.01, 0.01);
    assert_close_bf16(&actual_key, &expected_key, 0.01, 0.01);
    Ok(())
}

#[test]
fn rope_qk_bf16_correctness() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    for (num_tokens, num_q_heads, num_kv_heads, head_dim) in [
        (1, 1, 1, 2),
        (3, 2, 1, 4),
        (4, 4, 2, 8),
        (3, 8, 2, 32),
        (4, 32, 8, 64),
        (1, 2, 1, 512),
    ] {
        run_case(&runtime, num_tokens, num_q_heads, num_kv_heads, head_dim)?;
    }
    Ok(())
}

#[test]
fn rope_position_zero_is_identity() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    const NUM_TOKENS: usize = 1;
    const NUM_Q_HEADS: usize = 4;
    const NUM_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 64;

    let query_host = make_values(NUM_TOKENS * NUM_Q_HEADS * HEAD_DIM, 37, 257);
    let key_host = make_values(NUM_TOKENS * NUM_KV_HEADS * HEAD_DIM, 17, 193);
    let inv_freq = make_inv_freq(HEAD_DIM, 1_000_000.0);
    let position_ids = [0u32];
    let mut query = runtime.upload(&query_host, Shape::new([NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM]))?;
    let mut key = runtime.upload(&key_host, Shape::new([NUM_TOKENS, NUM_KV_HEADS, HEAD_DIM]))?;
    let inv_freq = runtime.upload(&inv_freq, Shape::new([HEAD_DIM / 2]))?;
    let position_ids = runtime.upload(&position_ids, Shape::new([NUM_TOKENS]))?;

    unsafe {
        runtime.kernels().rope().launch_qk_bf16_inplace(
            runtime.stream(),
            RopeLaunch {
                query: query.storage_mut(),
                key: key.storage_mut(),
                inv_freq: inv_freq.storage(),
                position_ids: position_ids.storage(),
                num_tokens: NUM_TOKENS,
                num_q_heads: NUM_Q_HEADS,
                num_kv_heads: NUM_KV_HEADS,
                head_dim: HEAD_DIM,
            },
        )?;
    }

    let actual_query = readback(&runtime, &query)?;
    let actual_key = readback(&runtime, &key)?;
    assert_eq_bf16(&actual_query, &query_host);
    assert_eq_bf16(&actual_key, &key_host);
    Ok(())
}

#[test]
fn rope_qk_bf16_long_positions() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    const NUM_TOKENS: usize = 4;
    const NUM_Q_HEADS: usize = 32;
    const NUM_KV_HEADS: usize = 8;
    const HEAD_DIM: usize = 64;

    let positions = [1u32, 1024, 16_384, 65_535];
    let query_host = make_values(NUM_TOKENS * NUM_Q_HEADS * HEAD_DIM, 37, 257);
    let key_host = make_values(NUM_TOKENS * NUM_KV_HEADS * HEAD_DIM, 17, 193);
    let inv_freq_host = make_inv_freq(HEAD_DIM, 1_000_000.0);
    let mut expected_query = query_host.clone();
    let mut expected_key = key_host.clone();
    rope_reference_inplace(
        &mut expected_query,
        &inv_freq_host,
        &positions,
        NUM_Q_HEADS,
        HEAD_DIM,
    );
    rope_reference_inplace(
        &mut expected_key,
        &inv_freq_host,
        &positions,
        NUM_KV_HEADS,
        HEAD_DIM,
    );

    let mut query = runtime.upload(&query_host, Shape::new([NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM]))?;
    let mut key = runtime.upload(&key_host, Shape::new([NUM_TOKENS, NUM_KV_HEADS, HEAD_DIM]))?;
    let inv_freq = runtime.upload(&inv_freq_host, Shape::new([HEAD_DIM / 2]))?;
    let positions = runtime.upload(&positions, Shape::new([NUM_TOKENS]))?;

    unsafe {
        runtime.kernels().rope().launch_qk_bf16_inplace(
            runtime.stream(),
            RopeLaunch {
                query: query.storage_mut(),
                key: key.storage_mut(),
                inv_freq: inv_freq.storage(),
                position_ids: positions.storage(),
                num_tokens: NUM_TOKENS,
                num_q_heads: NUM_Q_HEADS,
                num_kv_heads: NUM_KV_HEADS,
                head_dim: HEAD_DIM,
            },
        )?;
    }

    let actual_query = readback(&runtime, &query)?;
    let actual_key = readback(&runtime, &key)?;
    assert_close_bf16(&actual_query, &expected_query, 0.02, 0.02);
    assert_close_bf16(&actual_key, &expected_key, 0.02, 0.02);
    Ok(())
}

#[test]
#[ignore = "GPU benchmark"]
fn bench_rope_qk_bf16() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    const NUM_Q_HEADS: usize = 32;
    const NUM_KV_HEADS: usize = 8;
    const HEAD_DIM: usize = 64;

    println!("block_size={}", runtime.kernels().rope().block_size());
    let inv_freq_host = make_inv_freq(HEAD_DIM, 1_000_000.0);
    let inv_freq = runtime.upload(&inv_freq_host, Shape::new([HEAD_DIM / 2]))?;

    for num_tokens in [1, 4, 16, 64, 256, 1024, 4096] {
        let query_numel = num_tokens * NUM_Q_HEADS * HEAD_DIM;
        let key_numel = num_tokens * NUM_KV_HEADS * HEAD_DIM;
        let query_host = make_values(query_numel, 37, 257);
        let key_host = make_values(key_numel, 17, 193);
        let position_host: Vec<u32> = (0..num_tokens).map(|token| token as u32).collect();
        let mut query =
            runtime.upload(&query_host, Shape::new([num_tokens, NUM_Q_HEADS, HEAD_DIM]))?;
        let mut key =
            runtime.upload(&key_host, Shape::new([num_tokens, NUM_KV_HEADS, HEAD_DIM]))?;
        let positions = runtime.upload(&position_host, Shape::new([num_tokens]))?;

        let stats = benchmark_gpu(
            runtime.context(),
            runtime.stream(),
            BenchConfig::default(),
            || {
                unsafe {
                    runtime.kernels().rope().launch_qk_bf16_inplace(
                        runtime.stream(),
                        RopeLaunch {
                            query: query.storage_mut(),
                            key: key.storage_mut(),
                            inv_freq: inv_freq.storage(),
                            position_ids: positions.storage(),
                            num_tokens,
                            num_q_heads: NUM_Q_HEADS,
                            num_kv_heads: NUM_KV_HEADS,
                            head_dim: HEAD_DIM,
                        },
                    )?;
                }
                Ok(())
            },
        )?;

        let qk_elements = num_tokens * (NUM_Q_HEADS + NUM_KV_HEADS) * HEAD_DIM;
        let logical_bytes = qk_elements as f64 * 4.0;
        let seconds = stats.mean_us * 1e-6;
        let bandwidth = logical_bytes / seconds / 1e9;
        let tokens_per_second = num_tokens as f64 / seconds;
        println!(
            "tokens={num_tokens:>5} | mean={:>8.3} us | p50={:>8.3} us | p95={:>8.3} us | min={:>8.3} us | qk_io={:>8.2} GB/s | {:>12.0} tok/s",
            stats.mean_us, stats.p50_us, stats.p95_us, stats.min_us, bandwidth, tokens_per_second,
        );
    }
    Ok(())
}
