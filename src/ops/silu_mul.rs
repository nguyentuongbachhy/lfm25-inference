use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cuda::CudaRuntime,
    tensor::{Shape, Tensor},
};

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
    let mut output_dims = packed.dims().to_vec();
    let last = output_dims.len() - 1;
    output_dims[last] = intermediate_size;
    let mut out = runtime.alloc_bf16(Shape::new(output_dims))?;
    silu_mul_packed_bf16_into(runtime, packed, &mut out)?;
    Ok(out)
}

pub(crate) fn silu_mul_packed_bf16_into(
    runtime: &CudaRuntime,
    packed: &Tensor<bf16>,
    out: &mut Tensor<bf16>,
) -> Result<()> {
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
    out.set_logical_shape(Shape::new(output_dims))?;
    unsafe {
        runtime.kernels().silu_mul().launch_packed_bf16(
            runtime.stream(),
            packed.storage(),
            out.storage_mut(),
            rows,
            intermediate_size,
        )?;
    }
    Ok(())
}

pub(crate) fn silu_mul_packed_bf16_to_e4m3_into(
    runtime: &CudaRuntime,
    packed: &Tensor<bf16>,
    out: &mut Tensor<u8>,
    scale: f32,
) -> Result<()> {
    ensure!(
        packed.rank() >= 2,
        "packed silu_mul FP8 fusion expects rank >= 2, got {:?}",
        packed.dims()
    );
    ensure!(
        scale.is_finite() && scale > 0.0,
        "packed silu_mul FP8 fusion requires a finite positive scale"
    );
    let packed_width = packed.dims()[packed.rank() - 1];
    ensure!(
        packed_width > 0 && packed_width.is_multiple_of(2),
        "packed silu_mul FP8 fusion last dimension must be positive and even"
    );
    let intermediate_size = packed_width / 2;
    let rows = packed.numel() / packed_width;
    let mut output_dims = packed.dims().to_vec();
    let last = output_dims.len() - 1;
    output_dims[last] = intermediate_size;
    out.set_logical_shape(Shape::new(output_dims))?;
    unsafe {
        runtime.kernels().silu_mul().launch_packed_bf16_to_e4m3(
            runtime.stream(),
            packed.storage(),
            out.storage_mut(),
            rows,
            intermediate_size,
            scale,
        )?;
    }
    Ok(())
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

    #[test]
    fn fused_silu_mul_fp8_matches_unfused_pipeline_byte_for_byte() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        // Cover two rows, negative/positive values and an odd intermediate width
        // so both vectorized and scalar-tail semantics remain protected.
        let packed_host = [
            -4.0, -1.5, 0.25, 1.0, 2.5, 6.0,
            0.5, -2.0, 3.0, -0.75, 1.25, 4.0,
        ]
        .map(bf16::from_f32);
        let packed = runtime.upload(&packed_host, Shape::new([2, 6]))?;
        let scale = 9.75f32;

        let activated = silu_mul_packed_bf16(&runtime, &packed)?;
        let mut reference = runtime.alloc_fp8(activated.shape().clone())?;
        unsafe {
            runtime.kernels().fp8_quantize().launch_bf16_e4m3(
                runtime.stream(),
                activated.storage(),
                reference.storage_mut(),
                activated.numel(),
                scale,
            )?;
        }

        let mut fused = runtime.alloc_fp8(activated.shape().clone())?;
        silu_mul_packed_bf16_to_e4m3_into(&runtime, &packed, &mut fused, scale)?;

        let reference_host = runtime.download(&reference)?;
        let fused_host = runtime.download(&fused)?;
        assert_eq!(fused_host, reference_host);
        Ok(())
    }
}
