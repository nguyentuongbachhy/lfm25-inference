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
    let mut output = runtime.alloc_u32(Shape::new([rows]))?;
    argmax_rows_bf16_into(runtime, input, &mut output)?;
    Ok(output)
}

pub(crate) fn argmax_rows_bf16_into(
    runtime: &CudaRuntime,
    input: &Tensor<bf16>,
    output: &mut Tensor<u32>,
) -> Result<()> {
    ensure!(input.rank() == 2, "batched argmax expects rank-2 input");
    let rows = input.dims()[0];
    let columns = input.dims()[1];
    ensure!(rows > 0 && columns > 0, "batched argmax input is empty");
    output.set_logical_shape(Shape::new([rows]))?;
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

pub(crate) fn argmax_rows_bf16_atomic_into(
    runtime: &CudaRuntime,
    input: &Tensor<bf16>,
    output: &mut Tensor<u32>,
) -> Result<()> {
    ensure!(input.rank() == 2, "atomic argmax expects rank-2 input");
    let rows = input.dims()[0];
    let columns = input.dims()[1];
    ensure!(rows > 0 && columns > 0, "atomic argmax input is empty");
    output.set_logical_shape(Shape::new([rows]))?;
    unsafe {
        runtime.kernels().sampling().launch_argmax_rows_bf16_atomic(
            runtime.stream(),
            input.storage(),
            output.storage_mut(),
            rows,
            columns,
        )?;
    }
    Ok(())
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

    #[test]
    fn atomic_argmax_matches_legacy_tie_and_special_value_semantics() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let rows = 4usize;
        let columns = 1024usize;
        let mut host = vec![bf16::from_f32(-10.0); rows * columns];

        // Equal maxima in different legacy logical lanes: lane 0 wins even
        // though its absolute token index (256) is greater than token 1.
        host[1] = bf16::from_f32(7.0);
        host[256] = bf16::from_f32(7.0);

        // Equal maxima inside one legacy logical lane: earlier column wins.
        host[columns + 5] = bf16::from_f32(9.0);
        host[columns + 261] = bf16::from_f32(9.0);

        // +0 and -0 compare equal in the legacy float path; lane priority wins.
        let row2 = 2 * columns;
        host[row2 + 2] = bf16::from_bits(0x8000);
        host[row2 + 257] = bf16::from_bits(0x0000);
        for column in 0..columns {
            if column != 2 && column != 257 {
                host[row2 + column] = bf16::from_f32(-1.0);
            }
        }

        // NaN and -inf never beat legacy -FLT_MAX. An all-ignored row falls
        // through to index zero.
        let row3 = 3 * columns;
        for column in 0..columns {
            host[row3 + column] = if column & 1 == 0 {
                bf16::from_bits(0x7fc1)
            } else {
                bf16::from_bits(0xff80)
            };
        }

        let input = runtime.upload(&host, Shape::new([rows, columns]))?;
        let mut legacy = runtime.alloc_u32(Shape::new([rows]))?;
        let mut atomic = runtime.alloc_u32(Shape::new([rows]))?;
        argmax_rows_bf16_into(&runtime, &input, &mut legacy)?;
        argmax_rows_bf16_atomic_into(&runtime, &input, &mut atomic)?;
        runtime.synchronize()?;

        let legacy = runtime.download(&legacy)?;
        let atomic = runtime.download(&atomic)?;
        assert_eq!(atomic, legacy);
        assert_eq!(legacy[0], 256u32);
        assert_eq!(legacy[1], 5u32);
        assert_eq!(legacy[3], 0u32);
        Ok(())
    }

    #[test]
    fn atomic_argmax_matches_legacy_for_lfm_vocab() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let columns = 65_536usize;

        for rows in [1usize, 2, 8, 16] {
            let host = (0..rows * columns)
                .map(|index| {
                    // Low-cardinality values intentionally create many ties and
                    // stress the exact legacy logical-lane priority semantics.
                    let row = index / columns;
                    let column = index % columns;
                    let bucket = (column * 17 + row * 97 + column / 251) % 2048;
                    bf16::from_f32(bucket as f32 / 32.0 - 32.0)
                })
                .collect::<Vec<_>>();
            let input = runtime.upload(&host, Shape::new([rows, columns]))?;
            let mut legacy = runtime.alloc_u32(Shape::new([rows]))?;
            let mut atomic = runtime.alloc_u32(Shape::new([rows]))?;
            argmax_rows_bf16_into(&runtime, &input, &mut legacy)?;
            argmax_rows_bf16_atomic_into(&runtime, &input, &mut atomic)?;
            runtime.synchronize()?;
            assert_eq!(runtime.download(&atomic)?, runtime.download(&legacy)?);
        }
        Ok(())
    }

    #[test]
    fn atomic_argmax_rejects_columns_above_packing_limit() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let columns = 65_537usize;
        let input = runtime.upload(
            &vec![bf16::from_f32(0.0); columns],
            Shape::new([1, columns]),
        )?;
        let mut output = runtime.alloc_u32(Shape::new([1]))?;
        assert!(argmax_rows_bf16_atomic_into(&runtime, &input, &mut output).is_err());
        Ok(())
    }
}
