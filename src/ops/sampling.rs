use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cuda::CudaRuntime,
    tensor::{Shape, Tensor},
};

const ARGMAX_MULTIBLOCK_PARTIALS_PER_ROW: usize = 256;

pub(crate) struct ArgmaxRowsWorkspace {
    partial_values: Tensor<f32>,
    partial_indices: Tensor<u32>,
    maximum_rows: usize,
}

impl ArgmaxRowsWorkspace {
    pub(crate) fn new(runtime: &CudaRuntime, maximum_rows: usize) -> Result<Self> {
        ensure!(maximum_rows > 0, "argmax workspace requires positive row capacity");
        Ok(Self {
            partial_values: runtime.alloc_uninit::<f32>(Shape::new([
                maximum_rows,
                ARGMAX_MULTIBLOCK_PARTIALS_PER_ROW,
            ]))?,
            partial_indices: runtime.alloc_uninit::<u32>(Shape::new([
                maximum_rows,
                ARGMAX_MULTIBLOCK_PARTIALS_PER_ROW,
            ]))?,
            maximum_rows,
        })
    }
}

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

pub(crate) fn argmax_rows_bf16_multiblock_into(
    runtime: &CudaRuntime,
    input: &Tensor<bf16>,
    workspace: &mut ArgmaxRowsWorkspace,
    output: &mut Tensor<u32>,
) -> Result<()> {
    ensure!(input.rank() == 2, "multi-block argmax expects rank-2 input");
    let rows = input.dims()[0];
    let columns = input.dims()[1];
    ensure!(rows > 0 && columns > 0, "multi-block argmax input is empty");
    ensure!(
        rows <= workspace.maximum_rows,
        "multi-block argmax rows exceed workspace capacity"
    );
    workspace.partial_values.set_logical_shape(Shape::new([
        rows,
        ARGMAX_MULTIBLOCK_PARTIALS_PER_ROW,
    ]))?;
    workspace.partial_indices.set_logical_shape(Shape::new([
        rows,
        ARGMAX_MULTIBLOCK_PARTIALS_PER_ROW,
    ]))?;
    output.set_logical_shape(Shape::new([rows]))?;
    unsafe {
        runtime.kernels().sampling().launch_argmax_rows_bf16_multiblock(
            runtime.stream(),
            input.storage(),
            workspace.partial_values.storage_mut(),
            workspace.partial_indices.storage_mut(),
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

    use crate::cuda::benchmark::{BenchConfig, benchmark_gpu_paired};

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
    fn multiblock_argmax_matches_legacy_tie_semantics() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let rows = 2usize;
        let columns = 1024usize;
        let mut host = vec![bf16::from_f32(-10.0); rows * columns];

        // Legacy logical lane 0 sees column 256 while logical lane 1 sees
        // column 1. Equal maxima therefore select column 256 because the old
        // final reduction gives lower thread/lane id priority, not lower token
        // index priority.
        host[1] = bf16::from_f32(7.0);
        host[256] = bf16::from_f32(7.0);
        host[columns + 513] = bf16::from_f32(9.0);
        host[columns + 770] = bf16::from_f32(8.0);

        let input = runtime.upload(&host, Shape::new([rows, columns]))?;
        let mut legacy = runtime.alloc_u32(Shape::new([rows]))?;
        let mut candidate = runtime.alloc_u32(Shape::new([rows]))?;
        let mut workspace = ArgmaxRowsWorkspace::new(&runtime, rows)?;
        argmax_rows_bf16_into(&runtime, &input, &mut legacy)?;
        argmax_rows_bf16_multiblock_into(
            &runtime,
            &input,
            &mut workspace,
            &mut candidate,
        )?;
        runtime.synchronize()?;
        let legacy = runtime.download(&legacy)?;
        let candidate = runtime.download(&candidate)?;
        assert_eq!(legacy, [256u32, 513u32]);
        assert_eq!(candidate, legacy);
        Ok(())
    }

    #[test]
    fn multiblock_argmax_matches_legacy_for_lfm_vocab() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let columns = 65_536usize;
        for rows in [1usize, 2, 8] {
            let host = (0..rows * columns)
                .map(|index| {
                    // Deliberately low-cardinality BF16 values create many ties
                    // and stress the legacy logical-lane tie rule.
                    let row = index / columns;
                    let column = index % columns;
                    let bucket = (column * 17 + row * 97 + column / 251) % 2048;
                    bf16::from_f32(bucket as f32 / 32.0 - 32.0)
                })
                .collect::<Vec<_>>();
            let input = runtime.upload(&host, Shape::new([rows, columns]))?;
            let mut legacy = runtime.alloc_u32(Shape::new([rows]))?;
            let mut candidate = runtime.alloc_u32(Shape::new([rows]))?;
            let mut workspace = ArgmaxRowsWorkspace::new(&runtime, rows)?;
            argmax_rows_bf16_into(&runtime, &input, &mut legacy)?;
            argmax_rows_bf16_multiblock_into(
                &runtime,
                &input,
                &mut workspace,
                &mut candidate,
            )?;
            runtime.synchronize()?;
            assert_eq!(runtime.download(&candidate)?, runtime.download(&legacy)?);
        }
        Ok(())
    }

    #[test]
    #[ignore = "GPU microbenchmark"]
    fn bench_multiblock_argmax_lfm_vocab() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let columns = 65_536usize;
        let bench = BenchConfig {
            warmup: 16,
            batches: 40,
            iterations_per_batch: 20,
        };

        for rows in [1usize, 2, 4, 8, 16, 32, 64] {
            let host = (0..rows * columns)
                .map(|index| {
                    let row = index / columns;
                    let column = index % columns;
                    let bucket = (column * 29 + row * 131 + column / 113) % 8192;
                    bf16::from_f32(bucket as f32 / 64.0 - 64.0)
                })
                .collect::<Vec<_>>();
            let input = runtime.upload(&host, Shape::new([rows, columns]))?;
            let mut legacy_output = runtime.alloc_u32(Shape::new([rows]))?;
            let mut candidate_output = runtime.alloc_u32(Shape::new([rows]))?;
            let mut workspace = ArgmaxRowsWorkspace::new(&runtime, rows)?;

            argmax_rows_bf16_into(&runtime, &input, &mut legacy_output)?;
            argmax_rows_bf16_multiblock_into(
                &runtime,
                &input,
                &mut workspace,
                &mut candidate_output,
            )?;
            runtime.synchronize()?;
            assert_eq!(
                runtime.download(&candidate_output)?,
                runtime.download(&legacy_output)?
            );

            let stats = benchmark_gpu_paired(
                runtime.context(),
                runtime.stream(),
                bench,
                || argmax_rows_bf16_into(&runtime, &input, &mut legacy_output),
                || {
                    argmax_rows_bf16_multiblock_into(
                        &runtime,
                        &input,
                        &mut workspace,
                        &mut candidate_output,
                    )
                },
            )?;
            println!(
                "argmax_multiblock rows={} columns={} legacy_mean_us={:.3} candidate_mean_us={:.3} mean_speedup={:.4}x legacy_p95_us={:.3} candidate_p95_us={:.3} p95_speedup={:.4}x",
                rows,
                columns,
                stats.reference.mean_us,
                stats.candidate.mean_us,
                stats.speedup_mean,
                stats.reference.p95_us,
                stats.candidate.p95_us,
                stats.speedup_p95,
            );
        }
        Ok(())
    }
}
