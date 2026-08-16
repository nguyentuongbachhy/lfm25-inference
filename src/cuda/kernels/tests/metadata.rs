use anyhow::Result;

use crate::{
    cuda::{CudaRuntime, testing::readback},
    tensor::Shape,
};

fn append_u32(bytes: &mut Vec<u8>, values: &[u32]) {
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
}

fn append_i64(bytes: &mut Vec<u8>, values: &[i64]) {
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
}

#[test]
fn packed_metadata_scatter_preserves_all_fields() -> Result<()> {
    let runtime = CudaRuntime::new(0)?;
    let token_ids_host = [11u32, 22];
    let positions_host = [3u32, 7];
    let request_slots_host = [0u32, 1];
    let physical_slots_host = [5i64, 37];
    let segment_offsets_host = [0u32, 1, 2];
    let segment_slots_host = [0u32, 1];
    let output_rows_host = [0u32, 1];

    let mut packed_host = Vec::new();
    append_u32(&mut packed_host, &token_ids_host);
    append_u32(&mut packed_host, &positions_host);
    append_u32(&mut packed_host, &request_slots_host);
    while packed_host.len() % 8 != 0 {
        packed_host.push(0);
    }
    append_i64(&mut packed_host, &physical_slots_host);
    append_u32(&mut packed_host, &segment_offsets_host);
    append_u32(&mut packed_host, &segment_slots_host);
    append_u32(&mut packed_host, &output_rows_host);

    let packed = runtime.upload(&packed_host, Shape::new([packed_host.len()]))?;
    let mut token_ids = runtime.zeros::<u32>(Shape::new([2]))?;
    let mut positions = runtime.zeros::<u32>(Shape::new([2]))?;
    let mut request_slots = runtime.zeros::<u32>(Shape::new([2]))?;
    let mut physical_slots = runtime.zeros::<i64>(Shape::new([2]))?;
    let mut segment_offsets = runtime.zeros::<u32>(Shape::new([3]))?;
    let mut segment_slots = runtime.zeros::<u32>(Shape::new([2]))?;
    let mut output_rows = runtime.zeros::<u32>(Shape::new([2]))?;

    unsafe {
        runtime.kernels().metadata().launch_scatter(
            runtime.stream(),
            packed.storage(),
            token_ids.storage_mut(),
            positions.storage_mut(),
            request_slots.storage_mut(),
            physical_slots.storage_mut(),
            segment_offsets.storage_mut(),
            segment_slots.storage_mut(),
            output_rows.storage_mut(),
            2,
            2,
        )?;
    }
    runtime.synchronize()?;

    assert_eq!(readback(&runtime, &token_ids)?, token_ids_host);
    assert_eq!(readback(&runtime, &positions)?, positions_host);
    assert_eq!(readback(&runtime, &request_slots)?, request_slots_host);
    assert_eq!(readback(&runtime, &physical_slots)?, physical_slots_host);
    assert_eq!(readback(&runtime, &segment_offsets)?, segment_offsets_host);
    assert_eq!(readback(&runtime, &segment_slots)?, segment_slots_host);
    assert_eq!(readback(&runtime, &output_rows)?, output_rows_host);
    Ok(())
}
