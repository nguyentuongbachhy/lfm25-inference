use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cuda::{CudaRuntime, GatherLaunch},
    tensor::{Shape, Tensor},
};

pub fn gather_rows_bf16(
    runtime: &CudaRuntime,
    input: &Tensor<bf16>,
    rows: &Tensor<u32>,
) -> Result<Tensor<bf16>> {
    ensure!(input.rank() == 2, "row gather input must have rank 2");
    ensure!(
        rows.rank() == 1 && rows.numel() > 0,
        "row gather indices must be non-empty rank 1"
    );
    let mut output = runtime.alloc_bf16(Shape::new([rows.numel(), input.dims()[1]]))?;
    unsafe {
        runtime.kernels().gather().launch_rows_bf16(
            runtime.stream(),
            GatherLaunch {
                input: input.storage(),
                row_indices: rows.storage(),
                output: output.storage_mut(),
                output_rows: rows.numel(),
                input_rows: input.dims()[0],
                columns: input.dims()[1],
            },
        )?;
    }
    Ok(output)
}
