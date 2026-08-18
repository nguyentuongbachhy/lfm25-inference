use anyhow::Result;
use half::bf16;

use crate::{
    cuda::{
        CudaRuntime, KvCacheWriteLaunch,
        benchmark::{BenchConfig, benchmark_gpu},
        testing::{assert_eq_bf16, readback},
    },
    tensor::Shape,
};

fn raw_write(page_size: usize) -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let num_tokens = 3usize;
    let num_pages = 3usize;
    let source: Vec<bf16> = (0..num_tokens * 8 * 64)
        .map(|index| bf16::from_bits(index as u16 ^ 0x5a5a))
        .collect();
    let key = runtime.upload(&source, Shape::new([num_tokens, 8, 64]))?;
    let value = runtime.upload(&source, Shape::new([num_tokens, 8, 64]))?;
    let slots_host = [0i64, -1, (page_size * 2 + 3) as i64];
    let slots = runtime.upload(&slots_host, Shape::new([num_tokens]))?;
    let shape = Shape::new([num_pages, 8, page_size, 64]);
    let mut key_cache = runtime.zeros::<bf16>(shape.clone())?;
    let mut value_cache = runtime.zeros::<bf16>(shape)?;

    unsafe {
        runtime.kernels().kv_cache().launch_write_lfm2_bf16(
            runtime.stream(),
            KvCacheWriteLaunch {
                page_size,
                key: key.storage(),
                value: value.storage(),
                key_cache: key_cache.storage_mut(),
                value_cache: value_cache.storage_mut(),
                slot_mapping: slots.storage(),
                num_tokens,
                num_pages,
            },
        )?;
    }
    let actual = readback(&runtime, &key_cache)?;
    for token in [0usize, 2] {
        let slot = slots_host[token] as usize;
        for head in 0..8 {
            let source_start = (token * 8 + head) * 64;
            let destination = ((slot / page_size * 8 + head) * page_size + slot % page_size) * 64;
            assert_eq_bf16(
                &actual[destination..destination + 64],
                &source[source_start..source_start + 64],
            );
        }
    }
    Ok(())
}

#[test]
fn kv_cache_raw_correctness_ps16() -> Result<()> {
    raw_write(16)
}

#[test]
fn kv_cache_raw_correctness_ps32() -> Result<()> {
    raw_write(32)
}

#[test]
#[ignore = "GPU benchmark"]
fn bench_kv_cache_write_lfm2_bf16() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let config = BenchConfig {
        warmup: 50,
        batches: 50,
        iterations_per_batch: 100,
    };

    for page_size in [16usize, 32] {
        for num_tokens in [1usize, 4, 16, 64, 256, 1024, 4096] {
            let num_pages = num_tokens.div_ceil(page_size).max(1);
            let source = vec![bf16::from_f32(1.0); num_tokens * 8 * 64];
            let slots_host: Vec<i64> = (0..num_tokens).map(|slot| slot as i64).collect();
            let key = runtime.upload(&source, Shape::new([num_tokens, 8, 64]))?;
            let value = runtime.upload(&source, Shape::new([num_tokens, 8, 64]))?;
            let slots = runtime.upload(&slots_host, Shape::new([num_tokens]))?;
            let shape = Shape::new([num_pages, 8, page_size, 64]);
            let mut key_cache = runtime.zeros::<bf16>(shape.clone())?;
            let mut value_cache = runtime.zeros::<bf16>(shape)?;

            let stats = benchmark_gpu(runtime.context(), runtime.stream(), config, || {
                unsafe {
                    runtime.kernels().kv_cache().launch_write_lfm2_bf16(
                        runtime.stream(),
                        KvCacheWriteLaunch {
                            page_size,
                            key: key.storage(),
                            value: value.storage(),
                            key_cache: key_cache.storage_mut(),
                            value_cache: value_cache.storage_mut(),
                            slot_mapping: slots.storage(),
                            num_tokens,
                            num_pages,
                        },
                    )?;
                }
                Ok(())
            })?;
            println!(
                "page_size={page_size} tokens={num_tokens} mean={:.3}us p50={:.3}us p95={:.3}us min={:.3}us",
                stats.mean_us, stats.p50_us, stats.p95_us, stats.min_us,
            );
        }
    }
    Ok(())
}
