use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cuda::{CudaRuntime, Fp8LinearConfig, Fp8ScaleMode},
    model::quantization::ScalarScale,
    tensor::{Shape, Tensor},
};

pub(crate) fn quantize_weight_e4m3(
    runtime: &CudaRuntime,
    weight: &Tensor<bf16>,
    scale: ScalarScale,
) -> Result<Tensor<u8>> {
    ensure!(weight.numel() > 0, "cannot quantize an empty weight");
    let mut output = runtime.alloc_uninit::<u8>(weight.shape().clone())?;
    unsafe {
        runtime.kernels().fp8_quantize().launch_bf16_e4m3(
            runtime.stream(),
            weight.storage(),
            output.storage_mut(),
            weight.numel(),
            scale.quantize_multiplier,
        )?;
    }
    Ok(output)
}

pub(crate) fn linear_fp8_e4m3(
    runtime: &CudaRuntime,
    x: &Tensor<bf16>,
    weight: &Tensor<u8>,
    activation_scale: ScalarScale,
    weight_scale: ScalarScale,
) -> Result<Tensor<bf16>> {
    ensure!(x.rank() >= 1, "FP8 linear input must have rank >= 1");
    ensure!(weight.rank() == 2, "FP8 linear weight must have rank 2");
    ensure!(
        x.numel() > 0 && weight.numel() > 0,
        "FP8 linear does not support empty tensors"
    );
    let k = x.dims()[x.rank() - 1];
    let n = weight.dims()[0];
    ensure!(weight.dims()[1] == k, "FP8 linear K mismatch");
    let m = x.numel() / k;
    let mut output_dims = x.dims().to_vec();
    let last = output_dims.len() - 1;
    output_dims[last] = n;

    let mut quantized_input = runtime.alloc_fp8(x.shape().clone())?;
    unsafe {
        runtime.kernels().fp8_quantize().launch_bf16_e4m3(
            runtime.stream(),
            x.storage(),
            quantized_input.storage_mut(),
            x.numel(),
            activation_scale.quantize_multiplier,
        )?;
    }
    let mut output = runtime.alloc_bf16(Shape::new(output_dims))?;
    unsafe {
        runtime.blaslt().linear_fp8_scaled(
            quantized_input.storage(),
            weight.storage(),
            output.storage_mut(),
            Fp8LinearConfig {
                m,
                n,
                k,
                scale_mode: Fp8ScaleMode::Tensorwide,
                output_scale: activation_scale.dequantize_multiplier
                    * weight_scale.dequantize_multiplier,
            },
        )?;
    }
    Ok(output)
}

pub fn linear_bf16(
    runtime: &CudaRuntime,
    x: &Tensor<bf16>,
    weight: &Tensor<bf16>,
) -> Result<Tensor<bf16>> {
    ensure!(x.rank() >= 1, "linear input must have rank >= 1");
    ensure!(
        weight.rank() == 2,
        "linear weight must have rank 2, got {:?}",
        weight.dims()
    );
    let n = weight.dims()[0];
    let mut output_dims = x.dims().to_vec();
    let last = output_dims.len() - 1;
    output_dims[last] = n;
    let mut out = runtime.alloc_bf16(Shape::new(output_dims))?;
    linear_bf16_into(runtime, x, weight, &mut out)?;
    Ok(out)
}

pub(crate) fn linear_bf16_into(
    runtime: &CudaRuntime,
    x: &Tensor<bf16>,
    weight: &Tensor<bf16>,
    out: &mut Tensor<bf16>,
) -> Result<()> {
    ensure!(x.rank() >= 1, "linear input must have rank >= 1");
    ensure!(
        weight.rank() == 2,
        "linear weight must have rank 2, got {:?}",
        weight.dims()
    );
    ensure!(x.numel() > 0, "linear does not support empty input");
    ensure!(weight.numel() > 0, "linear does not support empty weight");
    let k = x.dims()[x.rank() - 1];
    let n = weight.dims()[0];
    let weight_k = weight.dims()[1];
    ensure!(
        k == weight_k,
        "linear dimension mismatch: input K={k}, weight={:?}",
        weight.dims()
    );
    let m = x.numel() / k;
    let mut output_dims = x.dims().to_vec();
    let last = output_dims.len() - 1;
    output_dims[last] = n;
    out.set_logical_shape(Shape::new(output_dims))?;
    unsafe {
        runtime
            .blaslt()
            .linear_bf16(x.storage(), weight.storage(), out.storage_mut(), m, n, k)?;
    }
    Ok(())
}

pub fn linear_last_row_bf16(
    runtime: &CudaRuntime,
    x: &Tensor<bf16>,
    weight: &Tensor<bf16>,
) -> Result<Tensor<bf16>> {
    ensure!(
        x.rank() == 2,
        "linear_last_row input must have rank 2, got {:?}",
        x.dims()
    );
    ensure!(
        weight.rank() == 2,
        "linear_last_row weight must have rank 2, got {:?}",
        weight.dims()
    );
    let rows = x.dims()[0];
    let k = x.dims()[1];
    let n = weight.dims()[0];
    ensure!(
        rows > 0 && k > 0 && n > 0,
        "linear_last_row does not support empty tensors"
    );
    ensure!(
        weight.dims()[1] == k,
        "linear_last_row dimension mismatch: input K={k}, weight={:?}",
        weight.dims()
    );
    let start = (rows - 1)
        .checked_mul(k)
        .ok_or_else(|| anyhow::anyhow!("linear_last_row offset overflow"))?;
    let end = start
        .checked_add(k)
        .ok_or_else(|| anyhow::anyhow!("linear_last_row range overflow"))?;
    let last_row = x
        .storage()
        .try_slice(start..end)
        .ok_or_else(|| anyhow::anyhow!("linear_last_row storage range is invalid"))?;
    let mut out = runtime.alloc_bf16(Shape::new([1, n]))?;
    unsafe {
        runtime
            .blaslt()
            .linear_bf16(&last_row, weight.storage(), out.storage_mut(), 1, n, k)?;
    }
    Ok(out)
}

pub(crate) fn linear_last_row_fp8_e4m3(
    runtime: &CudaRuntime,
    x: &Tensor<bf16>,
    weight: &Tensor<u8>,
    activation_scale: ScalarScale,
    weight_scale: ScalarScale,
) -> Result<Tensor<bf16>> {
    ensure!(x.rank() == 2, "FP8 last-row input must have rank 2");
    ensure!(weight.rank() == 2, "FP8 last-row weight must have rank 2");
    let rows = x.dims()[0];
    let k = x.dims()[1];
    let n = weight.dims()[0];
    ensure!(
        rows > 0 && k > 0 && weight.dims()[1] == k,
        "FP8 last-row shape mismatch"
    );
    let start = (rows - 1)
        .checked_mul(k)
        .ok_or_else(|| anyhow::anyhow!("FP8 last-row offset overflow"))?;
    let end = start
        .checked_add(k)
        .ok_or_else(|| anyhow::anyhow!("FP8 last-row range overflow"))?;
    let last_row = x
        .storage()
        .try_slice(start..end)
        .ok_or_else(|| anyhow::anyhow!("FP8 last-row storage range is invalid"))?;
    let mut quantized_input = runtime.alloc_fp8(Shape::new([1, k]))?;
    unsafe {
        runtime.kernels().fp8_quantize().launch_bf16_e4m3_view(
            runtime.stream(),
            &last_row,
            quantized_input.storage_mut(),
            k,
            activation_scale.quantize_multiplier,
        )?;
    }
    let mut output = runtime.alloc_bf16(Shape::new([1, n]))?;
    unsafe {
        runtime.blaslt().linear_fp8_scaled(
            quantized_input.storage(),
            weight.storage(),
            output.storage_mut(),
            Fp8LinearConfig {
                m: 1,
                n,
                k,
                scale_mode: Fp8ScaleMode::Tensorwide,
                output_scale: activation_scale.dequantize_multiplier
                    * weight_scale.dequantize_multiplier,
            },
        )?;
    }
    Ok(output)
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

    #[test]
    fn linear_bf16_preserves_prefix_shape() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let x_host = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].map(bf16::from_f32);
        let weight_host = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0].map(bf16::from_f32);
        let x = runtime.upload(&x_host, Shape::new([2, 2, 2]))?;
        let weight = runtime.upload(&weight_host, Shape::new([3, 2]))?;
        let out = linear_bf16(&runtime, &x, &weight)?;
        assert_eq!(out.dims(), &[2, 2, 3]);
        let expected =
            [1.0, 2.0, 3.0, 3.0, 4.0, 7.0, 5.0, 6.0, 11.0, 7.0, 8.0, 15.0].map(bf16::from_f32);
        assert_close_bf16(&readback(&runtime, &out)?, &expected, 0.01, 0.01);
        Ok(())
    }

    #[test]
    fn linear_rejects_k_mismatch() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let x = runtime.upload(&[bf16::from_f32(1.0); 8], Shape::new([2, 4]))?;
        let weight = runtime.upload(&[bf16::from_f32(1.0); 15], Shape::new([3, 5]))?;
        assert!(linear_bf16(&runtime, &x, &weight).is_err());
        Ok(())
    }

    #[test]
    fn linear_last_row_uses_input_storage_without_copy() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let x = runtime.upload(
            &[1.0, 2.0, 10.0, 20.0].map(bf16::from_f32),
            Shape::new([2, 2]),
        )?;
        let weight = runtime.upload(
            &[1.0, 0.0, 0.0, 1.0].map(bf16::from_f32),
            Shape::new([2, 2]),
        )?;
        let out = linear_last_row_bf16(&runtime, &x, &weight)?;
        assert_eq!(out.dims(), &[1, 2]);
        assert_close_bf16(
            &readback(&runtime, &out)?,
            &[10.0, 20.0].map(bf16::from_f32),
            0.01,
            0.01,
        );
        Ok(())
    }
}
