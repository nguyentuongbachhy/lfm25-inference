use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cuda::{CudaRuntime, ResidualRmsNormFp8Launch, ResidualRmsNormLaunch, RmsNormLaunch},
    tensor::Tensor,
};

pub fn rms_norm_bf16(
    runtime: &CudaRuntime,
    x: &Tensor<bf16>,
    weight: &Tensor<bf16>,
    eps: f32,
) -> Result<Tensor<bf16>> {
    let mut out = runtime.alloc_bf16(x.shape().clone())?;
    rms_norm_bf16_into(runtime, x, weight, eps, &mut out)?;
    Ok(out)
}

pub(crate) fn rms_norm_bf16_into(
    runtime: &CudaRuntime,
    x: &Tensor<bf16>,
    weight: &Tensor<bf16>,
    eps: f32,
    out: &mut Tensor<bf16>,
) -> Result<()> {
    ensure!(x.rank() >= 1, "RMSNorm input must have rank >= 1");
    ensure!(
        weight.rank() == 1,
        "RMSNorm weight must have rank 1, got {:?}",
        weight.dims()
    );
    ensure!(x.numel() > 0, "RMSNorm does not support empty tensors");
    ensure!(
        eps >= 0.0 && eps.is_finite(),
        "RMSNorm epsilon must be finite and non-negative"
    );
    let hidden_size = x.dims()[x.rank() - 1];
    ensure!(hidden_size > 0, "RMSNorm hidden size must be > 0");
    ensure!(
        weight.numel() == hidden_size,
        "RMSNorm weight mismatch: hidden_size={}, weight={:?}",
        hidden_size,
        weight.dims()
    );
    let rows = x.numel() / hidden_size;
    out.set_logical_shape(x.shape().clone())?;
    unsafe {
        runtime.kernels().rms_norm().launch_bf16(
            runtime.stream(),
            RmsNormLaunch {
                x: x.storage(),
                weight: weight.storage(),
                out: out.storage_mut(),
                rows,
                hidden_size,
                eps,
            },
        )?;
    }
    Ok(())
}

pub fn residual_rms_norm_bf16(
    runtime: &CudaRuntime,
    residual: &Tensor<bf16>,
    update: &Tensor<bf16>,
    weight: &Tensor<bf16>,
    eps: f32,
) -> Result<(Tensor<bf16>, Tensor<bf16>)> {
    let mut residual_out = runtime.alloc_bf16(residual.shape().clone())?;
    let mut normalized_out = runtime.alloc_bf16(residual.shape().clone())?;
    residual_rms_norm_bf16_into(
        runtime,
        residual,
        update,
        weight,
        eps,
        &mut residual_out,
        &mut normalized_out,
    )?;
    Ok((residual_out, normalized_out))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn residual_rms_norm_bf16_into(
    runtime: &CudaRuntime,
    residual: &Tensor<bf16>,
    update: &Tensor<bf16>,
    weight: &Tensor<bf16>,
    eps: f32,
    residual_out: &mut Tensor<bf16>,
    normalized_out: &mut Tensor<bf16>,
) -> Result<()> {
    ensure!(
        residual.shape() == update.shape(),
        "fused residual RMSNorm shape mismatch: residual={:?}, update={:?}",
        residual.dims(),
        update.dims()
    );
    ensure!(
        residual.rank() >= 1 && residual.numel() > 0,
        "fused residual RMSNorm requires a non-empty input"
    );
    ensure!(
        eps >= 0.0 && eps.is_finite(),
        "fused residual RMSNorm epsilon must be finite and non-negative"
    );
    let hidden_size = residual.dims()[residual.rank() - 1];
    ensure!(
        weight.rank() == 1 && weight.numel() == hidden_size,
        "fused residual RMSNorm weight mismatch: hidden_size={hidden_size}, weight={:?}",
        weight.dims()
    );
    let rows = residual.numel() / hidden_size;
    residual_out.set_logical_shape(residual.shape().clone())?;
    normalized_out.set_logical_shape(residual.shape().clone())?;
    unsafe {
        runtime.kernels().rms_norm().launch_residual_bf16(
            runtime.stream(),
            ResidualRmsNormLaunch {
                residual: residual.storage(),
                update: update.storage(),
                weight: weight.storage(),
                residual_out: residual_out.storage_mut(),
                normalized_out: normalized_out.storage_mut(),
                rows,
                hidden_size,
                eps,
            },
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn residual_rms_norm_bf16_to_e4m3_into(
    runtime: &CudaRuntime,
    residual: &Tensor<bf16>,
    update: &Tensor<bf16>,
    weight: &Tensor<bf16>,
    eps: f32,
    quant_scale: f32,
    residual_out: &mut Tensor<bf16>,
    normalized_fp8_out: &mut Tensor<u8>,
) -> Result<()> {
    ensure!(
        residual.shape() == update.shape(),
        "fused residual RMSNorm shape mismatch: residual={:?}, update={:?}",
        residual.dims(),
        update.dims()
    );
    ensure!(
        residual.rank() >= 1 && residual.numel() > 0,
        "fused residual RMSNorm requires a non-empty input"
    );
    ensure!(
        eps >= 0.0 && eps.is_finite(),
        "fused residual RMSNorm epsilon must be finite and non-negative"
    );
    let hidden_size = residual.dims()[residual.rank() - 1];
    ensure!(
        weight.rank() == 1 && weight.numel() == hidden_size,
        "fused residual RMSNorm weight mismatch: hidden_size={hidden_size}, weight={:?}",
        weight.dims()
    );
    ensure!(
        quant_scale.is_finite() && quant_scale > 0.0,
        "residual RMSNorm FP8 quantization scale must be finite and positive"
    );
    let rows = residual.numel() / hidden_size;
    residual_out.set_logical_shape(residual.shape().clone())?;
    normalized_fp8_out.set_logical_shape(residual.shape().clone())?;
    unsafe {
        runtime.kernels().residual_rms_fp8().launch(
            runtime.stream(),
            ResidualRmsNormFp8Launch {
                residual: residual.storage(),
                update: update.storage(),
                weight: weight.storage(),
                residual_out: residual_out.storage_mut(),
                normalized_fp8_out: normalized_fp8_out.storage_mut(),
                rows,
                hidden_size,
                eps,
                quant_scale,
            },
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod fused_tests {
    use super::*;
    use crate::{
        cuda::testing::{assert_close_bf16, readback},
        tensor::Shape,
    };
    use anyhow::Result;

    #[test]
    fn fused_residual_rms_norm_matches_cpu_reference() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let residual_host = [1.0, -2.0, 3.0, -4.0, 0.5, 1.5, -2.5, 3.5];
        let update_host = [0.5, 1.0, -1.0, 2.0, -0.5, 0.5, 1.5, -1.5];
        let weight_host = [1.0, 0.5, 1.5, 2.0];
        let residual = runtime.upload(&residual_host.map(bf16::from_f32), Shape::new([2, 4]))?;
        let update = runtime.upload(&update_host.map(bf16::from_f32), Shape::new([2, 4]))?;
        let weight = runtime.upload(&weight_host.map(bf16::from_f32), Shape::new([4]))?;
        let (sum, normalized) =
            residual_rms_norm_bf16(&runtime, &residual, &update, &weight, 1e-6)?;
        let expected_sum: Vec<bf16> = residual_host
            .iter()
            .zip(update_host)
            .map(|(left, right)| bf16::from_f32(left + right))
            .collect();
        let mut expected_normalized = Vec::with_capacity(8);
        for row in expected_sum.chunks_exact(4) {
            let variance = row.iter().map(|value| value.to_f32().powi(2)).sum::<f32>() / 4.0;
            let inv_rms = (variance + 1e-6).sqrt().recip();
            expected_normalized.extend(
                row.iter()
                    .zip(weight_host)
                    .map(|(value, scale)| bf16::from_f32(value.to_f32() * inv_rms * scale)),
            );
        }
        assert_eq!(readback(&runtime, &sum)?, expected_sum);
        assert_close_bf16(
            &readback(&runtime, &normalized)?,
            &expected_normalized,
            0.02,
            0.02,
        );
        Ok(())
    }

    #[test]
    fn residual_rms_norm_fp8_matches_two_kernel_reference_exactly() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let rows = 2usize;
        let hidden = 2048usize;
        let residual_host = (0..rows * hidden)
            .map(|index| bf16::from_f32(((index * 17 % 257) as f32 - 128.0) / 64.0))
            .collect::<Vec<_>>();
        let update_host = (0..rows * hidden)
            .map(|index| bf16::from_f32(((index * 23 % 251) as f32 - 125.0) / 96.0))
            .collect::<Vec<_>>();
        let weight_host = (0..hidden)
            .map(|index| bf16::from_f32(0.75 + (index * 7 % 31) as f32 / 64.0))
            .collect::<Vec<_>>();
        let residual = runtime.upload(&residual_host, Shape::new([rows, hidden]))?;
        let update = runtime.upload(&update_host, Shape::new([rows, hidden]))?;
        let weight = runtime.upload(&weight_host, Shape::new([hidden]))?;
        let mut reference_sum = runtime.alloc_bf16(Shape::new([rows, hidden]))?;
        let mut reference_norm = runtime.alloc_bf16(Shape::new([rows, hidden]))?;
        residual_rms_norm_bf16_into(
            &runtime,
            &residual,
            &update,
            &weight,
            1e-5,
            &mut reference_sum,
            &mut reference_norm,
        )?;
        let quant_scale = 153.3262f32;
        let mut reference_fp8 = runtime.alloc_fp8(Shape::new([rows, hidden]))?;
        unsafe {
            runtime.kernels().fp8_quantize().launch_bf16_e4m3(
                runtime.stream(),
                reference_norm.storage(),
                reference_fp8.storage_mut(),
                reference_norm.numel(),
                quant_scale,
            )?;
        }

        let mut fused_sum = runtime.alloc_bf16(Shape::new([rows, hidden]))?;
        let mut fused_fp8 = runtime.alloc_fp8(Shape::new([rows, hidden]))?;
        residual_rms_norm_bf16_to_e4m3_into(
            &runtime,
            &residual,
            &update,
            &weight,
            1e-5,
            quant_scale,
            &mut fused_sum,
            &mut fused_fp8,
        )?;

        assert_eq!(readback(&runtime, &fused_sum)?, readback(&runtime, &reference_sum)?);
        assert_eq!(readback(&runtime, &fused_fp8)?, readback(&runtime, &reference_fp8)?);
        Ok(())
    }
}
