use anyhow::Result;
use half::bf16;

use crate::{
    cache::{KvPageSize, PagedKvCache},
    cuda::{
        CudaRuntime,
        benchmark::{BenchConfig, benchmark_gpu_paired},
    },
    tensor::Shape,
};

use super::{
    qk_norm_rope_kv_write_decode_bf16, rms_norm_bf16, rope_qk_bf16_inplace,
};

const EPS: f32 = 1.0e-5;

fn bf16_values(elements: usize, mul: usize, modulus: usize, center: f32, scale: f32) -> Vec<bf16> {
    (0..elements)
        .map(|index| bf16::from_f32(((index * mul % modulus) as f32 - center) / scale))
        .collect()
}

fn norm_values(elements: usize, mul: usize) -> Vec<bf16> {
    (0..elements)
        .map(|index| bf16::from_f32(0.75 + ((index * mul % 29) as f32) / 64.0))
        .collect()
}

fn inv_freq_values() -> Vec<f32> {
    (0..32)
        .map(|index| 10_000.0f32.powf(-2.0 * index as f32 / 64.0))
        .collect()
}

#[test]
#[ignore = "GPU benchmark"]
fn bench_mok_paged_attention_paired_ab() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let query = runtime.upload(
        &vec![bf16::from_f32(0.01); 32 * 64],
        Shape::new([1, 32, 64]),
    )?;
    let config = BenchConfig {
        warmup: 20,
        batches: 60,
        iterations_per_batch: 20,
    };

    for page_size in [KvPageSize::P16, KvPageSize::P32] {
        for sequence_length in [16usize, 32, 128, 512, 2048, 8192] {
            let cache = PagedKvCache::new(&runtime, sequence_length, page_size)?;
            let position = runtime.upload(&[(sequence_length - 1) as u32], Shape::new([1]))?;
            let mut sync_output = runtime.alloc_bf16(Shape::new([1, 32, 64]))?;
            let mut async_output = runtime.alloc_bf16(Shape::new([1, 32, 64]))?;

            let paired = benchmark_gpu_paired(
                runtime.context(),
                runtime.stream(),
                config,
                || {
                    unsafe {
                        runtime.kernels().attention().launch_lfm2_bf16(
                            runtime.stream(),
                            page_size.value(),
                            query.storage(),
                            cache.key().storage(),
                            cache.value().storage(),
                            cache.block_table().storage(),
                            position.storage(),
                            sync_output.storage_mut(),
                            1,
                            cache.num_pages(),
                        )?;
                    }
                    Ok(())
                },
                || {
                    unsafe {
                        runtime.kernels().attention_async().launch_lfm2_bf16(
                            runtime.stream(),
                            page_size.value(),
                            query.storage(),
                            cache.key().storage(),
                            cache.value().storage(),
                            cache.block_table().storage(),
                            position.storage(),
                            async_output.storage_mut(),
                            1,
                            cache.num_pages(),
                        )?;
                    }
                    Ok(())
                },
            )?;

            println!(
                "mok_attention page_size={} context={} sync_mean={:.3}us sync_p50={:.3}us sync_p95={:.3}us async_mean={:.3}us async_p50={:.3}us async_p95={:.3}us paired_speedup_mean={:.4}x paired_speedup_p50={:.4}x speedup_min={:.4}x speedup_max={:.4}x",
                page_size.value(),
                sequence_length,
                paired.reference.mean_us,
                paired.reference.p50_us,
                paired.reference.p95_us,
                paired.candidate.mean_us,
                paired.candidate.p50_us,
                paired.candidate.p95_us,
                paired.speedup_mean,
                paired.speedup_p50,
                paired.speedup_min,
                paired.speedup_max,
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "GPU benchmark"]
fn bench_mok_qk_postprocess_paired_ab() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let query_host = bf16_values(32 * 64, 17, 101, 50.0, 64.0);
    let key_host = bf16_values(8 * 64, 13, 89, 44.0, 64.0);
    let value_host = bf16_values(8 * 64, 7, 79, 39.0, 32.0);
    let query_raw = runtime.upload(&query_host, Shape::new([1, 32, 64]))?;
    let key_raw = runtime.upload(&key_host, Shape::new([1, 8, 64]))?;
    let value = runtime.upload(&value_host, Shape::new([1, 8, 64]))?;
    let query_norm = runtime.upload(&norm_values(64, 3), Shape::new([64]))?;
    let key_norm = runtime.upload(&norm_values(64, 5), Shape::new([64]))?;
    let inv_freq = runtime.upload(&inv_freq_values(), Shape::new([32]))?;
    let position = runtime.upload(&[2047u32], Shape::new([1]))?;
    let config = BenchConfig {
        warmup: 20,
        batches: 60,
        iterations_per_batch: 20,
    };

    for page_size in [KvPageSize::P16, KvPageSize::P32] {
        let size = page_size.value();
        let slot = runtime.upload(&[i64::try_from(size - 1)?], Shape::new([1]))?;
        let mut reference_cache = PagedKvCache::new(&runtime, size, page_size)?;
        let mut fused_cache = PagedKvCache::new(&runtime, size, page_size)?;
        let mut fused_query = runtime.upload(&query_host, Shape::new([1, 32, 64]))?;

        // Prime the pool size classes used by the reference multi-kernel path.
        let mut warm_query = rms_norm_bf16(&runtime, &query_raw, &query_norm, EPS)?;
        let mut warm_key = rms_norm_bf16(&runtime, &key_raw, &key_norm, EPS)?;
        rope_qk_bf16_inplace(
            &runtime,
            &mut warm_query,
            &mut warm_key,
            &inv_freq,
            &position,
        )?;
        reference_cache.write_lfm2(&runtime, &warm_key, &value, &slot)?;
        drop(warm_query);
        drop(warm_key);
        runtime.synchronize()?;

        let paired = benchmark_gpu_paired(
            runtime.context(),
            runtime.stream(),
            config,
            || {
                let mut query = rms_norm_bf16(&runtime, &query_raw, &query_norm, EPS)?;
                let mut key = rms_norm_bf16(&runtime, &key_raw, &key_norm, EPS)?;
                rope_qk_bf16_inplace(
                    &runtime,
                    &mut query,
                    &mut key,
                    &inv_freq,
                    &position,
                )?;
                reference_cache.write_lfm2(&runtime, &key, &value, &slot)
            },
            || {
                qk_norm_rope_kv_write_decode_bf16(
                    &runtime,
                    &mut fused_query,
                    &key_raw,
                    &value,
                    &query_norm,
                    &key_norm,
                    &inv_freq,
                    &position,
                    &slot,
                    &mut fused_cache,
                    EPS,
                )
            },
        )?;

        println!(
            "mok_qk_postprocess page_size={} reference_mean={:.3}us reference_p50={:.3}us reference_p95={:.3}us fused_mean={:.3}us fused_p50={:.3}us fused_p95={:.3}us paired_speedup_mean={:.4}x paired_speedup_p50={:.4}x speedup_min={:.4}x speedup_max={:.4}x",
            page_size.value(),
            paired.reference.mean_us,
            paired.reference.p50_us,
            paired.reference.p95_us,
            paired.candidate.mean_us,
            paired.candidate.p50_us,
            paired.candidate.p95_us,
            paired.speedup_mean,
            paired.speedup_p50,
            paired.speedup_min,
            paired.speedup_max,
        );
    }
    Ok(())
}
