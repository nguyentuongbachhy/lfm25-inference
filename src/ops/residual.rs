#![allow(dead_code)]

use anyhow::{Result, ensure};
use half::bf16;

use crate::{cuda::CudaRuntime, tensor::Tensor};

pub fn residual_add_bf16(
    runtime: &CudaRuntime,
    residual: &Tensor<bf16>,
    update: &Tensor<bf16>,
) -> Result<Tensor<bf16>> {
    ensure!(
        residual.shape() == update.shape(),
        "residual shape mismatch: residual={:?}, update={:?}",
        residual.dims(),
        update.dims()
    );
    ensure!(
        residual.numel() > 0,
        "residual add does not support empty tensors"
    );
    let mut output = runtime.alloc_bf16(residual.shape().clone())?;
    unsafe {
        runtime.kernels().residual().launch_add_bf16(
            runtime.stream(),
            residual.storage(),
            update.storage(),
            output.storage_mut(),
            residual.numel(),
        )?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::{cuda::testing::assert_eq_bf16, tensor::Shape};

    use super::*;

    #[test]
    fn residual_add_handles_vector_body_and_scalar_tail() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let residual_host = [1.0, 2.0, 3.0, 4.0, 5.0].map(bf16::from_f32);
        let update_host = [5.0, 4.0, 3.0, 2.0, 1.0].map(bf16::from_f32);
        let residual = runtime.upload(&residual_host, Shape::new([5]))?;
        let update = runtime.upload(&update_host, Shape::new([5]))?;
        let output = residual_add_bf16(&runtime, &residual, &update)?;
        assert_eq_bf16(&runtime.download(&output)?, &[bf16::from_f32(6.0); 5]);
        Ok(())
    }
}
