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
    FusedAttentionInput, FusedPagedAttentionInput, QkPostprocessInput,
    fused_paged_attention_decode_lfm2_bf16, paged_attention_lfm2_bf16_sync,
    qk_norm_rope_kv_write_decode_bf16,
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
fn bench_mok_one_kernel_decode_attention_paired_ab() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let query_host = bf16_values(32 * 64, 17, 101, 50.0, 64.0);
    let key_host = bf16_values(8 * 64, 13, 89, 44.0, 64.0);
    let value_host = bf16_values(8 * 64, 7, 79, 39.0, 32.0);
    let query_raw = runtime.upload(&query_host, Shape::new([1, 32, 64]))?;
    let key_raw = runtime.upload(&key_host, Shape::new([1, 8, 64]))?;
    let value_raw = runtime.upload(&value_host, Shape::new([1, 8, 64]))?;
    let query_norm = runtime.upload(&norm_values(64, 3), Shape::new([64]))?;
    let key_norm = runtime.upload(&norm_values(64, 5), Shape::new([64]))?;
    let inv_freq = runtime.upload(&inv_freq_values(), Shape::new([32]))?;
    let config = BenchConfig {
        warmup: 20,
        batches: 60,
        iterations_per_batch: 20,
    };

    for page_size in [KvPageSize::P16, KvPageSize::P32] {
        for context in [16usize, 32, 128, 512, 2048, 8192] {
            let position = runtime.upload(&[u32::try_from(context - 1)?], Shape::new([1]))?;
            let slot = runtime.upload(&[i64::try_from(context - 1)?], Shape::new([1]))?;
            let mut reference_cache = PagedKvCache::new(&runtime, context, page_size)?;
            let mut candidate_cache = PagedKvCache::new(&runtime, context, page_size)?;
            let mut reference_query = runtime.upload(&query_host, Shape::new([1, 32, 64]))?;

            qk_norm_rope_kv_write_decode_bf16(
                &runtime,
                QkPostprocessInput {
                    query: &mut reference_query,
                    key: &key_raw,
                    value: &value_raw,
                    query_norm: &query_norm,
                    key_norm: &key_norm,
                    inv_freq: &inv_freq,
                    position_ids: &position,
                    slot_mapping: &slot,
                    eps: EPS,
                },
                &mut reference_cache,
            )?;
            drop(paged_attention_lfm2_bf16_sync(
                &runtime,
                &reference_query,
                &reference_cache,
                &position,
            )?);
            drop(fused_paged_attention_decode_lfm2_bf16(
                &runtime,
                FusedPagedAttentionInput {
                    attention: FusedAttentionInput {
                        query_raw: &query_raw,
                        key_raw: &key_raw,
                        value_raw: &value_raw,
                        query_norm: &query_norm,
                        key_norm: &key_norm,
                        inv_freq: &inv_freq,
                        position_ids: &position,
                        slot_mapping: &slot,
                        eps: EPS,
                    },
                    cache: &mut candidate_cache,
                },
            )?);
            runtime.synchronize()?;

            let paired = benchmark_gpu_paired(
                runtime.context(),
                runtime.stream(),
                config,
                || {
                    qk_norm_rope_kv_write_decode_bf16(
                        &runtime,
                        QkPostprocessInput {
                            query: &mut reference_query,
                            key: &key_raw,
                            value: &value_raw,
                            query_norm: &query_norm,
                            key_norm: &key_norm,
                            inv_freq: &inv_freq,
                            position_ids: &position,
                            slot_mapping: &slot,
                            eps: EPS,
                        },
                        &mut reference_cache,
                    )?;
                    drop(paged_attention_lfm2_bf16_sync(
                        &runtime,
                        &reference_query,
                        &reference_cache,
                        &position,
                    )?);
                    Ok(())
                },
                || {
                    drop(fused_paged_attention_decode_lfm2_bf16(
                        &runtime,
                        FusedPagedAttentionInput {
                            attention: FusedAttentionInput {
                                query_raw: &query_raw,
                                key_raw: &key_raw,
                                value_raw: &value_raw,
                                query_norm: &query_norm,
                                key_norm: &key_norm,
                                inv_freq: &inv_freq,
                                position_ids: &position,
                                slot_mapping: &slot,
                                eps: EPS,
                            },
                            cache: &mut candidate_cache,
                        },
                    )?);
                    Ok(())
                },
            )?;

            println!(
                "mok_one_kernel page_size={} context={} two_kernel_mean={:.3}us two_kernel_p50={:.3}us two_kernel_p95={:.3}us one_kernel_mean={:.3}us one_kernel_p50={:.3}us one_kernel_p95={:.3}us paired_speedup_mean={:.4}x paired_speedup_p50={:.4}x paired_speedup_p95={:.4}x speedup_min={:.4}x speedup_max={:.4}x",
                page_size.value(),
                context,
                paired.reference.mean_us,
                paired.reference.p50_us,
                paired.reference.p95_us,
                paired.candidate.mean_us,
                paired.candidate.p50_us,
                paired.candidate.p95_us,
                paired.speedup_mean,
                paired.speedup_p50,
                paired.speedup_p95,
                paired.speedup_min,
                paired.speedup_max,
            );
        }
    }
    Ok(())
}
