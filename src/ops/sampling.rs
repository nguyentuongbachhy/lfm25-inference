use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cuda::CudaRuntime,
    tensor::{Shape, Tensor},
};

pub fn argmax_bf16(runtime: &CudaRuntime, input: &Tensor<bf16>) -> Result<Tensor<u32>> {
    ensure!(input.numel() > 0, "argmax does not support empty tensors");
    let mut output = runtime.alloc_uninit::<u32>(Shape::new([1]))?;
    unsafe {
        runtime.kernels().sampling().launch_argmax_bf16(
            runtime.stream(),
            input.storage(),
            output.storage_mut(),
            input.numel(),
        )?;
    }
    Ok(output)
}

pub fn argmax_rows_bf16(runtime: &CudaRuntime, input: &Tensor<bf16>) -> Result<Tensor<u32>> {
    ensure!(input.rank() == 2, "batched argmax expects rank-2 input");
    let rows = input.dims()[0];
    let columns = input.dims()[1];
    ensure!(rows > 0 && columns > 0, "batched argmax input is empty");
    let mut output = runtime.alloc_u32(Shape::new([rows]))?;
    unsafe {
        runtime.kernels().sampling().launch_argmax_rows_bf16(
            runtime.stream(),
            input.storage(),
            output.storage_mut(),
            rows,
            columns,
        )?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;

    #[test]
    fn sampling_kernel_selects_argmax() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let host = [-2.0, 4.0, 8.0].map(bf16::from_f32);
        let input = runtime.upload(&host, Shape::new([1, 3]))?;
        let index = argmax_bf16(&runtime, &input)?;
        assert_eq!(runtime.download(&index)?, [2u32]);
        Ok(())
    }

    #[test]
    fn batched_sampling_selects_argmax_per_row() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let host = [-2.0, 4.0, 8.0, 9.0, 3.0, 7.0].map(bf16::from_f32);
        let input = runtime.upload(&host, Shape::new([2, 3]))?;
        let indices = argmax_rows_bf16(&runtime, &input)?;
        assert_eq!(runtime.download(&indices)?, [2u32, 0]);
        Ok(())
    }
}
