use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cuda::{CudaRuntime, ResidualRmsNormLaunch, RmsNormLaunch},
    tensor::Tensor,
};

pub fn rms_norm_bf16(
    runtime: &CudaRuntime,
    x: &Tensor<bf16>,
    weight: &Tensor<bf16>,
    eps: f32,
) -> Result<Tensor<bf16>> {
    ensure!(x.rank() >= 1, "RMSNorm input must have rank >= 1");
    ensure!(weight.rank() == 1, "RMSNorm weight must have rank 1, got {:?}", weight.dims());
    ensure!(x.numel() > 0, "RMSNorm does not support empty tensors");
    ensure!(eps >= 0.0 && eps.is_finite(), "RMSNorm epsilon must be finite and non-negative");
    let hidden_size = x.dims()[x.rank() - 1];
    ensure!(hidden_size > 0, "RMSNorm hidden size must be > 0");
    ensure!(weight.numel() == hidden_size, "RMSNorm weight mismatch: hidden_size={}, weight={:?}", hidden_size, weight.dims());
    let rows = x.numel() / hidden_size;
    let mut out = runtime.alloc_bf16(x.shape().clone())?;
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
    Ok(out)
}

pub fn residual_rms_norm_bf16(
    runtime: &CudaRuntime,
    residual: &Tensor<bf16>,
    update: &Tensor<bf16>,
    weight: &Tensor<bf16>,
    eps: f32,
) -> Result<(Tensor<bf16>, Tensor<bf16>)> {
    ensure!(residual.shape() == update.shape(), "fused residual RMSNorm shape mismatch: residual={:?}, update={:?}", residual.dims(), update.dims());
    ensure!(residual.rank() >= 1 && residual.numel() > 0, "fused residual RMSNorm requires a non-empty input");
    ensure!(eps >= 0.0 && eps.is_finite(), "fused residual RMSNorm epsilon must be finite and non-negative");
    let hidden_size = residual.dims()[residual.rank() - 1];
    ensure!(weight.rank() == 1 && weight.numel() == hidden_size, "fused residual RMSNorm weight mismatch: hidden_size={hidden_size}, weight={:?}", weight.dims());
    let rows = residual.numel() / hidden_size;
    let mut residual_out = runtime.alloc_bf16(residual.shape().clone())?;
    let mut normalized_out = runtime.alloc_bf16(residual.shape().clone())?;
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
    Ok((residual_out, normalized_out))
}

#[cfg(test)]
mod fused_tests {
    use anyhow::Result;
    use crate::{cuda::testing::{assert_close_bf16, readback}, tensor::Shape};
    use super::*;

    #[test]
    fn fused_residual_rms_norm_matches_cpu_reference() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let residual_host = [1.0, -2.0, 3.0, -4.0, 0.5, 1.5, -2.5, 3.5];
        let update_host = [0.5, 1.0, -1.0, 2.0, -0.5, 0.5, 1.5, -1.5];
        let weight_host = [1.0, 0.5, 1.5, 2.0];
        let residual = runtime.upload(&residual_host.map(bf16::from_f32), Shape::new([2, 4]))?;
        let update = runtime.upload(&update_host.map(bf16::from_f32), Shape::new([2, 4]))?;
        let weight = runtime.upload(&weight_host.map(bf16::from_f32), Shape::new([4]))?;
        let (sum, normalized) = residual_rms_norm_bf16(&runtime, &residual, &update, &weight, 1e-6)?;
        let expected_sum: Vec<bf16> = residual_host.iter().zip(update_host).map(|(left, right)| bf16::from_f32(left + right)).collect();
        let mut expected_normalized = Vec::with_capacity(8);
        for row in expected_sum.chunks_exact(4) {
            let variance = row.iter().map(|value| value.to_f32().powi(2)).sum::<f32>() / 4.0;
            let inv_rms = (variance + 1e-6).sqrt().recip();
            expected_normalized.extend(row.iter().zip(weight_host).map(|(value, scale)| bf16::from_f32(value.to_f32() * inv_rms * scale)));
        }
        assert_eq!(readback(&runtime, &sum)?, expected_sum);
        assert_close_bf16(&readback(&runtime, &normalized)?, &expected_normalized, 0.02, 0.02);
        Ok(())
    }
}
