use anyhow::{Result, ensure};
use half::bf16;

use crate::{cuda::CudaRuntime, tensor::Tensor};

pub(crate) fn argmax_rows_bf16_into(
    runtime: &CudaRuntime,
    input: &Tensor<bf16>,
    output: &mut Tensor<u32>,
) -> Result<()> {
    ensure!(input.rank() == 2, "batched argmax expects rank-2 input");
    let rows = input.dims()[0];
    let columns = input.dims()[1];
    ensure!(rows > 0 && columns > 0, "batched argmax input is empty");
    ensure!(
        output.storage_capacity() >= rows,
        "persistent batched argmax output is too small"
    );
    unsafe {
        runtime.kernels().sampling().launch_argmax_rows_bf16(
            runtime.stream(),
            input.storage(),
            output.storage_mut(),
            rows,
            columns,
        )?;
    }
    Ok(())
}
