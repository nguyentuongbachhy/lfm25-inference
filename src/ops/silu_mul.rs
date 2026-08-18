use anyhow::{Result, ensure};
use half::bf16;

use crate::{cuda::CudaRuntime, tensor::Tensor};

pub fn silu_mul_packed_bf16(runtime: &CudaRuntime, packed: &Tensor<bf16>) -> Result<Tensor<bf16>> {
    ensure!(
        packed.rank() >= 2,
        "packed silu_mul expects rank >= 2, got {:?}",
        packed.dims()
    );
    let packed_width = packed.dims()[packed.rank() - 1];
    ensure!(
        packed_width > 0 && packed_width.is_multiple_of(2),
        "packed silu_mul last dimension must be positive and even"
    );
    let intermediate_size = packed_width / 2;
    let rows = packed.numel() / packed_width;
    let mut output_dims = packed.dims().to_vec();
    let last = output_dims.len() - 1;
    output_dims[last] = intermediate_size;
    let mut out = runtime.alloc_bf16(crate::tensor::Shape::new(output_dims))?;
    unsafe {
        runtime.kernels().silu_mul().launch_packed_bf16(
            runtime.stream(),
            packed.storage(),
            out.storage_mut(),
            rows,
            intermediate_size,
        )?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use half::bf16;

    use crate::{
        cuda::{
            CudaRuntime,
            testing::{assert_close_bf16, readback},
        },
        tensor::Shape,
    };

    use super::*;

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    #[test]
    fn packed_silu_mul_handles_multiple_rows() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let packed_host = [
            -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, -0.5, 2.0,
        ]
        .map(bf16::from_f32);
        let packed = runtime.upload(&packed_host, Shape::new([2, 6]))?;
        let out = silu_mul_packed_bf16(&runtime, &packed)?;
        let expected = [
            silu(-2.0) * 1.0,
            silu(-1.0) * 2.0,
            silu(0.0) * 3.0,
            silu(4.0) * 1.5,
            silu(5.0) * -0.5,
            silu(0.5) * 2.0,
        ]
        .map(bf16::from_f32);
        assert_close_bf16(&readback(&runtime, &out)?, &expected, 0.02, 0.02);
        Ok(())
    }
}
