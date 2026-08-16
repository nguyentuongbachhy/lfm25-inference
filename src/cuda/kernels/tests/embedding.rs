use anyhow::Result;
use half::bf16;

use crate::{
    cuda::{
        CudaRuntime,
        benchmark::{BenchConfig, benchmark_gpu},
        testing::{assert_eq_bf16, readback},
    },
    tensor::Shape,
};

fn make_weight(vocab_size: usize, hidden_size: usize) -> Vec<bf16> {
    (0..vocab_size * hidden_size)
        .map(|i| {
            let value = ((i % 2048) as f32 - 1024.0) / 512.0;

            bf16::from_f32(value)
        })
        .collect()
}

fn expected_embedding(
    token_ids: &[u32],
    weight: &[bf16],
    vocab_size: usize,
    hidden_size: usize,
) -> Vec<bf16> {
    let mut out = vec![bf16::from_f32(0.0); token_ids.len() * hidden_size];

    for (token_idx, &token_id) in token_ids.iter().enumerate() {
        let token_id = token_id as usize;

        if token_id >= vocab_size {
            continue;
        }

        let src_start = token_id * hidden_size;

        let src_end = src_start + hidden_size;

        let dst_start = token_idx * hidden_size;

        let dst_end = dst_start + hidden_size;

        out[dst_start..dst_end].copy_from_slice(&weight[src_start..src_end]);
    }

    out
}

fn run_embedding_case(
    runtime: &CudaRuntime,
    vocab_size: usize,
    hidden_size: usize,
    token_ids: &[u32],
) -> Result<()> {
    let weight_host = make_weight(vocab_size, hidden_size);

    let expected = expected_embedding(token_ids, &weight_host, vocab_size, hidden_size);

    let token_ids_gpu = runtime.upload(token_ids, Shape::new([token_ids.len()]))?;

    let weight_gpu = runtime.upload(&weight_host, Shape::new([vocab_size, hidden_size]))?;

    let mut out = runtime.zeros::<bf16>(Shape::new([token_ids.len(), hidden_size]))?;

    unsafe {
        runtime.kernels().embedding().launch_bf16(
            runtime.stream(),
            token_ids_gpu.storage(),
            weight_gpu.storage(),
            out.storage_mut(),
            token_ids.len(),
            vocab_size,
            hidden_size,
        )?;
    }

    let actual = readback(runtime, &out)?;

    assert_eq_bf16(&actual, &expected);

    Ok(())
}

#[test]
fn embedding_bf16_correctness() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;

    const VOCAB_SIZE: usize = 17;

    let token_ids = [0, 16, 1, 8, 3, 12, 4];

    for hidden_size in [
        1, 2, 3, 7, 8, 9, 31, 32, 33, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048,
        2049,
    ] {
        run_embedding_case(&runtime, VOCAB_SIZE, hidden_size, &token_ids)?;
    }

    Ok(())
}

#[test]
fn embedding_bf16_repeated_tokens() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;

    run_embedding_case(&runtime, 8, 128, &[3, 3, 3, 7, 7, 0, 3])
}

#[test]
fn embedding_bf16_invalid_token_is_zero() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;

    run_embedding_case(&runtime, 8, 64, &[0, 7, 8, 100, 3])
}

#[test]
#[ignore = "GPU benchmark"]
fn bench_embedding_bf16() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;

    const VOCAB_SIZE: usize = 65_536;

    const HIDDEN_SIZE: usize = 2_048;

    let weight_host = make_weight(VOCAB_SIZE, HIDDEN_SIZE);

    let weight = runtime.upload(&weight_host, Shape::new([VOCAB_SIZE, HIDDEN_SIZE]))?;

    for num_tokens in [1, 4, 16, 64, 256, 1024, 4096] {
        let token_ids: Vec<u32> = (0usize..num_tokens)
            .map(|i| (i.wrapping_mul(7_919) % VOCAB_SIZE) as u32)
            .collect();

        let token_ids_gpu = runtime.upload(&token_ids, Shape::new([num_tokens]))?;

        let mut out = runtime.zeros::<bf16>(Shape::new([num_tokens, HIDDEN_SIZE]))?;

        let stats = benchmark_gpu(
            runtime.context(),
            runtime.stream(),
            BenchConfig::default(),
            || {
                unsafe {
                    runtime.kernels().embedding().launch_bf16(
                        runtime.stream(),
                        token_ids_gpu.storage(),
                        weight.storage(),
                        out.storage_mut(),
                        num_tokens,
                        VOCAB_SIZE,
                        HIDDEN_SIZE,
                    )?;
                }

                Ok(())
            },
        )?;

        let bytes_per_token = HIDDEN_SIZE * size_of::<bf16>() * 2 + size_of::<u32>();

        let total_bytes = num_tokens as f64 * bytes_per_token as f64;

        let seconds = stats.mean_us * 1e-6;

        let bandwidth_gbps = total_bytes / seconds / 1e9;

        let tokens_per_second = num_tokens as f64 / seconds;

        println!(
            "tokens={num_tokens:>5} | \
             mean={:>8.3} us | \
             p50={:>8.3} us | \
             p95={:>8.3} us | \
             min={:>8.3} us | \
             {:>8.2} GB/s | \
             {:>12.0} tok/s",
            stats.mean_us,
            stats.p50_us,
            stats.p95_us,
            stats.min_us,
            bandwidth_gbps,
            tokens_per_second,
        );
    }

    Ok(())
}
