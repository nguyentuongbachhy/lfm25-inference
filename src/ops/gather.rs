use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cuda::CudaRuntime,
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
            input.storage(),
            rows.storage(),
            output.storage_mut(),
            rows.numel(),
            input.dims()[0],
            input.dims()[1],
        )?;
    }
    Ok(output)
}
