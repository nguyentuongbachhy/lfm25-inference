use anyhow::{Context as _, Result};
use half::bf16;

use crate::{
    cache::{KvPageSize, PagedKvArena},
    cuda::{
        CudaRuntime,
        testing::{assert_close_bf16, readback},
    },
    tensor::Shape,
};

use super::{
    attention::paged_ragged_attention_lfm2_bf16,
    attention_async_fast::{
        FastRaggedAttentionInput, paged_ragged_attention_splitk_lfm2_bf16_into,
        splitk_workspace_elements,
    },
};

fn compare_splitk_to_reference(page_size: KvPageSize) -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    // Different lengths intentionally exercise independent block-table rows.
    // With 8 splits, the shorter request also exercises empty split ranges.
    let lengths = [257usize, 65usize];
    let pages = lengths.map(|length| length.div_ceil(page_size.value()));
    let block_table_stride = *pages.iter().max().context("missing page count")?;
    let total_pages = pages.iter().sum::<usize>();
    let total_tokens = lengths.iter().sum::<usize>();

    let query_host = (0..lengths.len() * 32 * 64)
        .map(|index| bf16::from_f32(((index * 17 % 113) as f32 - 56.0) / 96.0))
        .collect::<Vec<_>>();
    let key_host = (0..total_tokens * 8 * 64)
        .map(|index| bf16::from_f32(((index * 13 % 97) as f32 - 48.0) / 96.0))
        .collect::<Vec<_>>();
    let value_host = (0..total_tokens * 8 * 64)
        .map(|index| bf16::from_f32(((index * 7 % 89) as f32 - 44.0) / 48.0))
        .collect::<Vec<_>>();

    let query = runtime.upload(&query_host, Shape::new([lengths.len(), 32, 64]))?;
    let key = runtime.upload(&key_host, Shape::new([total_tokens, 8, 64]))?;
    let value = runtime.upload(&value_host, Shape::new([total_tokens, 8, 64]))?;

    let mut block_tables = vec![u32::MAX; lengths.len() * block_table_stride];
    let mut physical_slots = Vec::with_capacity(total_tokens);
    let mut physical_page_base = 0usize;
    for (request, (&length, &request_pages)) in lengths.iter().zip(&pages).enumerate() {
        for logical_page in 0..request_pages {
            block_tables[request * block_table_stride + logical_page] =
                u32::try_from(physical_page_base + logical_page)?;
        }
        for position in 0..length {
            let physical_page = physical_page_base + position / page_size.value();
            let offset = position % page_size.value();
            let physical_slot = physical_page
                .checked_mul(page_size.value())
                .and_then(|base| base.checked_add(offset))
                .context("physical slot overflow")?;
            physical_slots.push(i64::try_from(physical_slot)?);
        }
        physical_page_base += request_pages;
    }
    assert_eq!(physical_page_base, total_pages);

    let physical_slots = runtime.upload(&physical_slots, Shape::new([total_tokens]))?;
    let mut arena = PagedKvArena::new(&runtime, total_pages, page_size)?;
    arena.write_lfm2(&runtime, &key, &value, &physical_slots)?;
    let block_tables = runtime.upload(
        &block_tables,
        Shape::new([lengths.len(), block_table_stride]),
    )?;
    let request_slots = runtime.upload(&[0u32, 1u32], Shape::new([2]))?;
    let positions = runtime.upload(
        &[
            u32::try_from(lengths[0] - 1)?,
            u32::try_from(lengths[1] - 1)?,
        ],
        Shape::new([2]),
    )?;

    // Use the independent reference paged-attention kernel here rather than the
    // optimized async-fast kernel. This prevents a shared fast-path bug from
    // making the split-K regression test pass accidentally.
    let reference = paged_ragged_attention_lfm2_bf16(
        &runtime,
        &query,
        &arena,
        &block_tables,
        block_table_stride,
        &request_slots,
        &positions,
    )?;
    let reference_host = readback(&runtime, &reference)?;

    let mut partials = runtime.alloc_uninit::<f32>(Shape::new([splitk_workspace_elements(2)?]))?;
    let mut candidate = runtime.alloc_bf16(Shape::new([2, 32, 64]))?;
    for splits in [2usize, 4, 8] {
        let input = FastRaggedAttentionInput {
            query: &query,
            arena: &arena,
            block_tables: &block_tables,
            block_table_stride,
            request_slots: &request_slots,
            position_ids: &positions,
        };
        paged_ragged_attention_splitk_lfm2_bf16_into(
            &runtime,
            input,
            &mut partials,
            splits,
            &mut candidate,
        )?;
        assert_close_bf16(
            &readback(&runtime, &candidate)?,
            &reference_host,
            0.04,
            0.03,
        );
    }
    Ok(())
}

#[test]
fn splitk_ragged_attention_matches_reference_ps16() -> Result<()> {
    compare_splitk_to_reference(KvPageSize::P16)
}

#[test]
fn splitk_ragged_attention_matches_reference_ps32() -> Result<()> {
    compare_splitk_to_reference(KvPageSize::P32)
}
