use anyhow::Result;
use half::bf16;

use crate::{
    cuda::{
        CudaRuntime, QkPackedPostprocessLaunch, QkPostprocessLaunch, QkvUnpackLaunch,
        testing::{assert_close_bf16, readback},
    },
    tensor::Shape,
};

const Q_WIDTH: usize = 32 * 64;
const KV_WIDTH: usize = 8 * 64;
const PACKED_WIDTH: usize = Q_WIDTH + 2 * KV_WIDTH;

fn bf16_values(elements: usize, mul: usize, modulus: usize, center: f32, scale: f32) -> Vec<bf16> {
    (0..elements)
        .map(|index| {
            bf16::from_f32(((index.wrapping_mul(mul) % modulus) as f32 - center) / scale)
        })
        .collect()
}

fn norm_values(elements: usize, mul: usize) -> Vec<bf16> {
    (0..elements)
        .map(|index| bf16::from_f32(0.75 + ((index * mul % 29) as f32) / 64.0))
        .collect()
}

#[test]
fn packed_qk_postprocess_matches_unpacked_path_exactly() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let tokens = 2usize;
    let page_size = 16usize;
    let num_pages = 2usize;
    let cache_elements = num_pages * 8 * page_size * 64;

    let packed = runtime.upload(
        &bf16_values(tokens * PACKED_WIDTH, 17, 257, 128.0, 128.0),
        Shape::new([tokens, PACKED_WIDTH]),
    )?;
    let query_norm = runtime.upload(&norm_values(64, 3), Shape::new([64]))?;
    let key_norm = runtime.upload(&norm_values(64, 5), Shape::new([64]))?;
    let inv_freq = runtime.upload(
        &(0..32)
            .map(|index| 10_000.0f32.powf(-2.0 * index as f32 / 64.0))
            .collect::<Vec<_>>(),
        Shape::new([32]),
    )?;
    let positions = runtime.upload(&[15u32, 16u32], Shape::new([tokens]))?;
    let slots = runtime.upload(&[15i64, 16i64], Shape::new([tokens]))?;

    let mut direct_query = runtime.alloc_uninit::<bf16>(Shape::new([tokens, 32, 64]))?;
    let mut direct_key = runtime.alloc_uninit::<bf16>(Shape::new([tokens, 8, 64]))?;
    let mut direct_value = runtime.alloc_uninit::<bf16>(Shape::new([tokens, 8, 64]))?;
    let mut direct_key_cache = runtime.zeros::<bf16>(Shape::new([cache_elements]))?;
    let mut direct_value_cache = runtime.zeros::<bf16>(Shape::new([cache_elements]))?;

    let mut packed_query = runtime.alloc_uninit::<bf16>(Shape::new([tokens, 32, 64]))?;
    let mut packed_key_cache = runtime.zeros::<bf16>(Shape::new([cache_elements]))?;
    let mut packed_value_cache = runtime.zeros::<bf16>(Shape::new([cache_elements]))?;

    unsafe {
        runtime.kernels().qkv_unpack().launch_bf16(
            runtime.stream(),
            QkvUnpackLaunch {
                packed: packed.storage(),
                query: direct_query.storage_mut(),
                key: direct_key.storage_mut(),
                value: direct_value.storage_mut(),
                num_tokens: tokens,
            },
        )?;
        runtime.kernels().qk_postprocess().launch_decode(
            runtime.stream(),
            QkPostprocessLaunch {
                page_size,
                query: direct_query.storage_mut(),
                key: direct_key.storage(),
                value: direct_value.storage(),
                query_norm: query_norm.storage(),
                key_norm: key_norm.storage(),
                inv_freq: inv_freq.storage(),
                position_ids: positions.storage(),
                slot_mapping: slots.storage(),
                key_cache: direct_key_cache.storage_mut(),
                value_cache: direct_value_cache.storage_mut(),
                num_tokens: tokens,
                num_pages,
                eps: 1.0e-5,
            },
        )?;
        runtime.kernels().qk_postprocess().launch_packed_decode(
            runtime.stream(),
            QkPackedPostprocessLaunch {
                page_size,
                packed_qkv: packed.storage(),
                query: packed_query.storage_mut(),
                query_norm: query_norm.storage(),
                key_norm: key_norm.storage(),
                inv_freq: inv_freq.storage(),
                position_ids: positions.storage(),
                slot_mapping: slots.storage(),
                key_cache: packed_key_cache.storage_mut(),
                value_cache: packed_value_cache.storage_mut(),
                num_tokens: tokens,
                num_pages,
                eps: 1.0e-5,
            },
        )?;
    }
    runtime.synchronize()?;

    assert_close_bf16(
        &readback(&runtime, &direct_query)?,
        &readback(&runtime, &packed_query)?,
        0.0,
        0.0,
    );
    assert_close_bf16(
        &readback(&runtime, &direct_key_cache)?,
        &readback(&runtime, &packed_key_cache)?,
        0.0,
        0.0,
    );
    assert_close_bf16(
        &readback(&runtime, &direct_value_cache)?,
        &readback(&runtime, &packed_value_cache)?,
        0.0,
        0.0,
    );
    Ok(())
}
