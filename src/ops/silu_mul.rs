use anyhow::{Result, ensure};
use half::bf16;

use crate::{cuda::CudaRuntime, tensor::Tensor};

#[allow(dead_code)]
pub fn silu_mul_bf16(
    runtime: &CudaRuntime,
    gate: &Tensor<bf16>,
    up: &Tensor<bf16>,
) -> Result<Tensor<bf16>> {
    ensure!(
        gate.shape() == up.shape(),
        "silu_mul shape mismatch: gate={:?}, up={:?}",
        gate.dims(),
        up.dims()
    );

    ensure!(gate.numel() > 0, "silu_mul does not support empty tensor",);

    let numel = gate.numel();
    let mut out = runtime.alloc_bf16(gate.shape().clone())?;

    unsafe {
        runtime.kernels().silu_mul().launch_bf16(
            runtime.stream(),
            gate.storage(),
            up.storage(),
            out.storage_mut(),
            numel,
        )?;
    }

    Ok(out)
}

pub fn silu_mul_packed_bf16(runtime: &CudaRuntime, packed: &Tensor<bf16>) -> Result<Tensor<bf16>> {
    ensure!(
        packed.rank() >= 2,
        "packed silu_mul expects rank >= 2, got {:?}",
        packed.dims()
    );
    let packed_width = packed.dims()[packed.rank() - 1];
    ensure!(
        packed_width > 0 && packed_width % 2 == 0,
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
    fn silu_mul_bf16_end_to_end() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;

        let gate_host = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0].map(bf16::from_f32);

        let up_host = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(bf16::from_f32);

        let gate = runtime.upload(&gate_host, Shape::new([2, 3]))?;

        let up = runtime.upload(&up_host, Shape::new([2, 3]))?;

        let out = silu_mul_bf16(&runtime, &gate, &up)?;

        assert_eq!(out.dims(), &[2, 3],);

        let actual = readback(&runtime, &out)?;

        let expected: Vec<bf16> = gate_host
            .iter()
            .zip(up_host.iter())
            .map(|(&gate, &up)| bf16::from_f32(silu(gate.to_f32()) * up.to_f32()))
            .collect();

        assert_close_bf16(&actual, &expected, 0.01, 0.01);

        Ok(())
    }

    #[test]
    fn silu_mul_rejects_shape_mismatch() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;

        let gate = runtime.upload(&[bf16::from_f32(1.0); 6], Shape::new([2, 3]))?;

        let up = runtime.upload(&[bf16::from_f32(1.0); 6], Shape::new([6]))?;

        assert!(silu_mul_bf16(&runtime, &gate, &up,).is_err());

        Ok(())
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
