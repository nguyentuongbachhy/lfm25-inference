use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cuda::{CudaRuntime, EmbeddingLaunch},
    tensor::{Shape, Tensor},
};

pub fn embedding_bf16(
    runtime: &CudaRuntime,
    token_ids: &Tensor<u32>,
    weight: &Tensor<bf16>,
) -> Result<Tensor<bf16>> {
    ensure!(
        weight.rank() == 2,
        "embedding weight must have rank 2, got shape {:?}",
        weight.dims()
    );
    ensure!(
        token_ids.numel() > 0,
        "embedding does not support empty token tensors"
    );

    let vocab_size = weight.dims()[0];
    let hidden_size = weight.dims()[1];
    ensure!(vocab_size > 0, "embedding vocab size must be greater than zero");
    ensure!(hidden_size > 0, "embedding hidden size must be greater than zero");

    let num_tokens = token_ids.numel();
    let mut output_dims = token_ids.dims().to_vec();
    output_dims.push(hidden_size);
    let mut out = runtime.alloc_bf16(Shape::new(output_dims))?;

    unsafe {
        runtime.kernels().embedding().launch_bf16(
            runtime.stream(),
            EmbeddingLaunch {
                token_ids: token_ids.storage(),
                weight: weight.storage(),
                out: out.storage_mut(),
                num_tokens,
                vocab_size,
                hidden_size,
            },
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
            testing::{assert_eq_bf16, readback},
        },
        tensor::Shape,
    };

    use super::*;

    #[test]
    fn embedding_bf16_rank1() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let weight_host = [
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
            bf16::from_f32(4.0),
            bf16::from_f32(5.0),
            bf16::from_f32(6.0),
            bf16::from_f32(7.0),
            bf16::from_f32(8.0),
            bf16::from_f32(9.0),
            bf16::from_f32(10.0),
            bf16::from_f32(11.0),
            bf16::from_f32(12.0),
        ];
        let weight = runtime.upload(&weight_host, Shape::new([4, 3]))?;
        let token_ids = runtime.upload(&[2u32, 0, 3, 1], Shape::new([4]))?;
        let out = embedding_bf16(&runtime, &token_ids, &weight)?;
        assert_eq!(out.dims(), &[4, 3]);
        let actual = readback(&runtime, &out)?;
        let expected = [
            7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 4.0, 5.0, 6.0,
        ]
        .map(bf16::from_f32);
        assert_eq_bf16(&actual, &expected);
        Ok(())
    }

    #[test]
    fn embedding_bf16_preserves_token_shape() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let weight_host: Vec<bf16> = (0..20).map(|i| bf16::from_f32(i as f32)).collect();
        let weight = runtime.upload(&weight_host, Shape::new([5, 4]))?;
        let token_ids = runtime.upload(&[4u32, 0, 2, 1, 3, 4], Shape::new([2, 3]))?;
        let out = embedding_bf16(&runtime, &token_ids, &weight)?;
        assert_eq!(out.dims(), &[2, 3, 4]);
        let actual = readback(&runtime, &out)?;
        let expected_ids = [4usize, 0, 2, 1, 3, 4];
        let mut expected = Vec::new();
        for token_id in expected_ids {
            let start = token_id * 4;
            expected.extend_from_slice(&weight_host[start..start + 4]);
        }
        assert_eq_bf16(&actual, &expected);
        Ok(())
    }

    #[test]
    fn embedding_rejects_invalid_weight_rank() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let token_ids = runtime.upload(&[0u32, 1], Shape::new([2]))?;
        let weight = runtime.upload(&[bf16::from_f32(1.0); 8], Shape::new([2, 2, 2]))?;
        assert!(embedding_bf16(&runtime, &token_ids, &weight).is_err());
        Ok(())
    }
}
