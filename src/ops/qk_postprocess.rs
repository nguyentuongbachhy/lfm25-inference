use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cache::{PagedKvArena, PagedKvCache},
    cuda::{CudaRuntime, QkPostprocessLaunch},
    tensor::Tensor,
};

pub(crate) struct QkPostprocessInput<'a> {
    pub(crate) query: &'a mut Tensor<bf16>,
    pub(crate) key: &'a Tensor<bf16>,
    pub(crate) value: &'a Tensor<bf16>,
    pub(crate) query_norm: &'a Tensor<bf16>,
    pub(crate) key_norm: &'a Tensor<bf16>,
    pub(crate) inv_freq: &'a Tensor<f32>,
    pub(crate) position_ids: &'a Tensor<u32>,
    pub(crate) slot_mapping: &'a Tensor<i64>,
    pub(crate) eps: f32,
}

fn launch_qk_postprocess(
    runtime: &CudaRuntime,
    input: QkPostprocessInput<'_>,
    key_cache: &mut Tensor<bf16>,
    value_cache: &mut Tensor<bf16>,
    page_size: usize,
    num_pages: usize,
) -> Result<()> {
    let QkPostprocessInput {
        query,
        key,
        value,
        query_norm,
        key_norm,
        inv_freq,
        position_ids,
        slot_mapping,
        eps,
    } = input;
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
    ensure!(
        query_norm.numel() == 64,
        "query norm weight must have 64 elements"
    );
    ensure!(
        key_norm.numel() == 64,
        "key norm weight must have 64 elements"
    );
    ensure!(
        inv_freq.numel() == 32,
        "RoPE inv_freq must have 32 elements"
    );
    ensure!(
        position_ids.numel() == num_tokens,
        "position count mismatch"
    );
    ensure!(
        slot_mapping.numel() == num_tokens,
        "slot mapping count mismatch"
    );

    unsafe {
        runtime.kernels().qk_postprocess().launch_decode(
            runtime.stream(),
            QkPostprocessLaunch {
                page_size,
                query: query.storage_mut(),
                key: key.storage(),
                value: value.storage(),
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
    Ok(())
}

pub(crate) fn qk_norm_rope_kv_write_decode_bf16(
    runtime: &CudaRuntime,
    input: QkPostprocessInput<'_>,
    cache: &mut PagedKvCache,
) -> Result<()> {
    let page_size = cache.page_size().value();
    let num_pages = cache.num_pages();
    let (key_cache, value_cache) = cache.kv_mut();
    launch_qk_postprocess(runtime, input, key_cache, value_cache, page_size, num_pages)
}

pub(crate) fn qk_norm_rope_kv_write_arena_decode_bf16(
    runtime: &CudaRuntime,
    input: QkPostprocessInput<'_>,
    arena: &mut PagedKvArena,
) -> Result<()> {
    let page_size = arena.page_size().value();
    let num_pages = arena.num_pages();
    let (key_cache, value_cache) = arena.kv_mut();
    launch_qk_postprocess(runtime, input, key_cache, value_cache, page_size, num_pages)
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::{
        cache::KvPageSize,
        cuda::{
            benchmark::{BenchConfig, benchmark_gpu},
            testing::{assert_close_bf16, readback},
        },
        ops::{rms_norm_bf16, rope_qk_bf16_inplace},
        tensor::Shape,
    };

    use super::*;

    const EPS: f32 = 1.0e-5;

    struct ReferenceInput<'a> {
        query_raw: &'a Tensor<bf16>,
        key_raw: &'a Tensor<bf16>,
        value: &'a Tensor<bf16>,
        query_norm: &'a Tensor<bf16>,
        key_norm: &'a Tensor<bf16>,
        inv_freq: &'a Tensor<f32>,
        positions: &'a Tensor<u32>,
        slots: &'a Tensor<i64>,
    }

    fn bf16_values(
        elements: usize,
        mul: usize,
        modulus: usize,
        center: f32,
        scale: f32,
    ) -> Vec<bf16> {
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

    fn run_reference(
        runtime: &CudaRuntime,
        input: ReferenceInput<'_>,
        cache: &mut PagedKvCache,
    ) -> Result<Tensor<bf16>> {
        let mut query = rms_norm_bf16(runtime, input.query_raw, input.query_norm, EPS)?;
        let mut key = rms_norm_bf16(runtime, input.key_raw, input.key_norm, EPS)?;
        rope_qk_bf16_inplace(
            runtime,
            &mut query,
            &mut key,
            input.inv_freq,
            input.positions,
        )?;
        cache.write_lfm2(runtime, &key, input.value, input.slots)?;
        Ok(query)
    }

    fn check_fused_matches_reference(page_size: KvPageSize) -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let size = page_size.value();
        let tokens = 2usize;
        let query_host = bf16_values(tokens * 32 * 64, 17, 101, 50.0, 64.0);
        let key_host = bf16_values(tokens * 8 * 64, 13, 89, 44.0, 64.0);
        let value_host = bf16_values(tokens * 8 * 64, 7, 79, 39.0, 32.0);
        let query_norm_host = norm_values(64, 3);
        let key_norm_host = norm_values(64, 5);
        let inv_freq_host = inv_freq_values();
        let positions_host = [u32::try_from(size - 1)?, u32::try_from(size)?];
        let slots_host = [i64::try_from(size - 1)?, i64::try_from(size)?];

        let query_raw = runtime.upload(&query_host, Shape::new([tokens, 32, 64]))?;
        let key_raw = runtime.upload(&key_host, Shape::new([tokens, 8, 64]))?;
        let value = runtime.upload(&value_host, Shape::new([tokens, 8, 64]))?;
        let query_norm = runtime.upload(&query_norm_host, Shape::new([64]))?;
        let key_norm = runtime.upload(&key_norm_host, Shape::new([64]))?;
        let inv_freq = runtime.upload(&inv_freq_host, Shape::new([32]))?;
        let positions = runtime.upload(&positions_host, Shape::new([tokens]))?;
        let slots = runtime.upload(&slots_host, Shape::new([tokens]))?;

        let mut reference_cache = PagedKvCache::new(&runtime, size * 2, page_size)?;
        let reference_query = run_reference(
            &runtime,
            ReferenceInput {
                query_raw: &query_raw,
                key_raw: &key_raw,
                value: &value,
                query_norm: &query_norm,
                key_norm: &key_norm,
                inv_freq: &inv_freq,
                positions: &positions,
                slots: &slots,
            },
            &mut reference_cache,
        )?;

        let mut fused_query = runtime.upload(&query_host, Shape::new([tokens, 32, 64]))?;
        let mut fused_cache = PagedKvCache::new(&runtime, size * 2, page_size)?;
        qk_norm_rope_kv_write_decode_bf16(
            &runtime,
            QkPostprocessInput {
                query: &mut fused_query,
                key: &key_raw,
                value: &value,
                query_norm: &query_norm,
                key_norm: &key_norm,
                inv_freq: &inv_freq,
                position_ids: &positions,
                slot_mapping: &slots,
                eps: EPS,
            },
            &mut fused_cache,
        )?;

        assert_close_bf16(
            &readback(&runtime, &fused_query)?,
            &readback(&runtime, &reference_query)?,
            0.03,
            0.02,
        );
        assert_close_bf16(
            &readback(&runtime, fused_cache.key())?,
            &readback(&runtime, reference_cache.key())?,
            0.03,
            0.02,
        );
        assert_close_bf16(
            &readback(&runtime, fused_cache.value())?,
            &readback(&runtime, reference_cache.value())?,
            0.0,
            0.0,
        );
        Ok(())
    }

    #[test]
    fn fused_qk_postprocess_ps16_matches_reference_across_page_boundary() -> Result<()> {
        check_fused_matches_reference(KvPageSize::P16)
    }

    #[test]
    fn fused_qk_postprocess_ps32_matches_reference_across_page_boundary() -> Result<()> {
        check_fused_matches_reference(KvPageSize::P32)
    }

    #[test]
    #[ignore = "GPU benchmark"]
    fn bench_qk_postprocess_fused_vs_reference_bf16() -> Result<()> {
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
        let config = BenchConfig {
            warmup: 30,
            batches: 40,
            iterations_per_batch: 25,
        };

        for page_size in [KvPageSize::P16, KvPageSize::P32] {
            let size = page_size.value();
            let position = runtime.upload(&[2047u32], Shape::new([1]))?;
            let slot = runtime.upload(&[i64::try_from(size - 1)?], Shape::new([1]))?;
            let mut reference_cache = PagedKvCache::new(&runtime, size, page_size)?;
            let mut fused_cache = PagedKvCache::new(&runtime, size, page_size)?;
            let mut fused_query = runtime.upload(&query_host, Shape::new([1, 32, 64]))?;

            let _ = run_reference(
                &runtime,
                ReferenceInput {
                    query_raw: &query_raw,
                    key_raw: &key_raw,
                    value: &value,
                    query_norm: &query_norm,
                    key_norm: &key_norm,
                    inv_freq: &inv_freq,
                    positions: &position,
                    slots: &slot,
                },
                &mut reference_cache,
            )?;
            runtime.synchronize()?;

            let reference = benchmark_gpu(runtime.context(), runtime.stream(), config, || {
                let _query = run_reference(
                    &runtime,
                    ReferenceInput {
                        query_raw: &query_raw,
                        key_raw: &key_raw,
                        value: &value,
                        query_norm: &query_norm,
                        key_norm: &key_norm,
                        inv_freq: &inv_freq,
                        positions: &position,
                        slots: &slot,
                    },
                    &mut reference_cache,
                )?;
                Ok(())
            })?;

            let fused = benchmark_gpu(runtime.context(), runtime.stream(), config, || {
                qk_norm_rope_kv_write_decode_bf16(
                    &runtime,
                    QkPostprocessInput {
                        query: &mut fused_query,
                        key: &key_raw,
                        value: &value,
                        query_norm: &query_norm,
                        key_norm: &key_norm,
                        inv_freq: &inv_freq,
                        position_ids: &position,
                        slot_mapping: &slot,
                        eps: EPS,
                    },
                    &mut fused_cache,
                )
            })?;

            println!(
                "page_size={} reference_mean={:.3}us reference_p50={:.3}us reference_p95={:.3}us fused_mean={:.3}us fused_p50={:.3}us fused_p95={:.3}us speedup={:.3}x",
                page_size.value(),
                reference.mean_us,
                reference.p50_us,
                reference.p95_us,
                fused.mean_us,
                fused.p50_us,
                fused.p95_us,
                reference.mean_us / fused.mean_us,
            );
        }
        Ok(())
    }
}
