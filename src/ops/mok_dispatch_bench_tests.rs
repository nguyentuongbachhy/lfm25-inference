use anyhow::Result;
use half::bf16;

use crate::{
    cache::{KvPageSize, PagedKvArena},
    cuda::{
        CudaRuntime,
        benchmark::{BenchConfig, benchmark_gpu_paired},
    },
    tensor::Shape,
};

use super::{
    fused_ragged_paged_attention_decode_lfm2_bf16,
    paged_ragged_attention_fast_lfm2_bf16,
    qk_norm_rope_kv_write_arena_decode_bf16,
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
fn bench_mok_short_dispatch_ragged_paired_ab() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let query_norm = runtime.upload(&norm_values(64, 3), Shape::new([64]))?;
    let key_norm = runtime.upload(&norm_values(64, 5), Shape::new([64]))?;
    let inv_freq = runtime.upload(&inv_freq_values(), Shape::new([32]))?;
    let config = BenchConfig {
        warmup: 12,
        batches: 36,
        iterations_per_batch: 12,
    };

    for page_size in [KvPageSize::P16, KvPageSize::P32] {
        let page = page_size.value();
        for context in [16usize, 32, 64, 128] {
            for batch in [1usize, 2, 4, 8, 16, 32, 64] {
                let pages_per_request = context.div_ceil(page);
                let total_pages = batch * pages_per_request;

                let mut block_tables_host = Vec::with_capacity(total_pages);
                for request in 0..batch {
                    let base = request * pages_per_request;
                    for logical_page in 0..pages_per_request {
                        block_tables_host.push(u32::try_from(base + logical_page)?);
                    }
                }
                let block_tables = runtime.upload(
                    &block_tables_host,
                    Shape::new([batch, pages_per_request]),
                )?;
                let request_slots_host = (0..batch)
                    .map(u32::try_from)
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let request_slots = runtime.upload(
                    &request_slots_host,
                    Shape::new([batch]),
                )?;
                let positions_host = vec![u32::try_from(context - 1)?; batch];
                let positions = runtime.upload(&positions_host, Shape::new([batch]))?;

                let history_per_request = context - 1;
                let history_tokens = batch * history_per_request;
                let mut history_slots_host = Vec::with_capacity(history_tokens);
                for request in 0..batch {
                    let base_page = request * pages_per_request;
                    for position in 0..history_per_request {
                        let physical_page = base_page + position / page;
                        let offset = position % page;
                        history_slots_host.push(i64::try_from(physical_page * page + offset)?);
                    }
                }
                let history_slots = runtime.upload(
                    &history_slots_host,
                    Shape::new([history_tokens]),
                )?;
                let history_key = runtime.upload(
                    &bf16_values(history_tokens * 8 * 64, 11, 83, 41.0, 64.0),
                    Shape::new([history_tokens, 8, 64]),
                )?;
                let history_value = runtime.upload(
                    &bf16_values(history_tokens * 8 * 64, 7, 79, 39.0, 32.0),
                    Shape::new([history_tokens, 8, 64]),
                )?;

                let mut current_slots_host = Vec::with_capacity(batch);
                for request in 0..batch {
                    let position = context - 1;
                    let physical_page = request * pages_per_request + position / page;
                    let offset = position % page;
                    current_slots_host.push(i64::try_from(physical_page * page + offset)?);
                }
                let current_slots = runtime.upload(
                    &current_slots_host,
                    Shape::new([batch]),
                )?;

                let query_host = bf16_values(batch * 32 * 64, 17, 101, 50.0, 64.0);
                let key_host = bf16_values(batch * 8 * 64, 13, 89, 44.0, 64.0);
                let value_host = bf16_values(batch * 8 * 64, 5, 73, 36.0, 32.0);
                let query_raw = runtime.upload(&query_host, Shape::new([batch, 32, 64]))?;
                let key_raw = runtime.upload(&key_host, Shape::new([batch, 8, 64]))?;
                let value_raw = runtime.upload(&value_host, Shape::new([batch, 8, 64]))?;
                let mut reference_query = runtime.upload(&query_host, Shape::new([batch, 32, 64]))?;

                let mut reference_arena = PagedKvArena::new(&runtime, total_pages, page_size)?;
                let mut fused_arena = PagedKvArena::new(&runtime, total_pages, page_size)?;
                reference_arena.write_lfm2(
                    &runtime,
                    &history_key,
                    &history_value,
                    &history_slots,
                )?;
                fused_arena.write_lfm2(
                    &runtime,
                    &history_key,
                    &history_value,
                    &history_slots,
                )?;

                qk_norm_rope_kv_write_arena_decode_bf16(
                    &runtime,
                    &mut reference_query,
                    &key_raw,
                    &value_raw,
                    &query_norm,
                    &key_norm,
                    &inv_freq,
                    &positions,
                    &current_slots,
                    &mut reference_arena,
                    EPS,
                )?;
                drop(paged_ragged_attention_fast_lfm2_bf16(
                    &runtime,
                    &reference_query,
                    &reference_arena,
                    &block_tables,
                    pages_per_request,
                    &request_slots,
                    &positions,
                )?);
                drop(fused_ragged_paged_attention_decode_lfm2_bf16(
                    &runtime,
                    &query_raw,
                    &key_raw,
                    &value_raw,
                    &query_norm,
                    &key_norm,
                    &inv_freq,
                    &block_tables,
                    pages_per_request,
                    &request_slots,
                    &positions,
                    &current_slots,
                    &mut fused_arena,
                    EPS,
                )?);
                runtime.synchronize()?;

                let paired = benchmark_gpu_paired(
                    runtime.context(),
                    runtime.stream(),
                    config,
                    || {
                        qk_norm_rope_kv_write_arena_decode_bf16(
                            &runtime,
                            &mut reference_query,
                            &key_raw,
                            &value_raw,
                            &query_norm,
                            &key_norm,
                            &inv_freq,
                            &positions,
                            &current_slots,
                            &mut reference_arena,
                            EPS,
                        )?;
                        drop(paged_ragged_attention_fast_lfm2_bf16(
                            &runtime,
                            &reference_query,
                            &reference_arena,
                            &block_tables,
                            pages_per_request,
                            &request_slots,
                            &positions,
                        )?);
                        Ok(())
                    },
                    || {
                        drop(fused_ragged_paged_attention_decode_lfm2_bf16(
                            &runtime,
                            &query_raw,
                            &key_raw,
                            &value_raw,
                            &query_norm,
                            &key_norm,
                            &inv_freq,
                            &block_tables,
                            pages_per_request,
                            &request_slots,
                            &positions,
                            &current_slots,
                            &mut fused_arena,
                            EPS,
                        )?);
                        Ok(())
                    },
                )?;

                println!(
                    "mok_dispatch page_size={} context={} batch={} two_kernel_fast_mean={:.3}us two_kernel_fast_p50={:.3}us two_kernel_fast_p95={:.3}us one_kernel_mean={:.3}us one_kernel_p50={:.3}us one_kernel_p95={:.3}us paired_speedup_mean={:.4}x paired_speedup_p50={:.4}x paired_speedup_p95={:.4}x speedup_min={:.4}x speedup_max={:.4}x",
                    page,
                    context,
                    batch,
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
    }
    Ok(())
}
