use anyhow::{Context as _, Result, ensure};
use half::bf16;

use crate::{
    cuda::{
        CudaRuntime, INT8_TINY_M_LIMIT, QuantizeS8RowsLaunch, TinyMInt8LinearLaunch,
    },
    tensor::{Shape, Tensor},
};

pub(crate) struct Int8PerChannelWeight {
    pub(crate) data: Tensor<i8>,
    pub(crate) scales: Tensor<f32>,
}

pub(crate) struct Int8TinyMWorkspace {
    quantized_input: Tensor<i8>,
    input_scales: Tensor<f32>,
    maximum_m: usize,
    k: usize,
}

impl Int8TinyMWorkspace {
    pub(crate) fn new(runtime: &CudaRuntime, maximum_m: usize, k: usize) -> Result<Self> {
        ensure!(
            (1..=INT8_TINY_M_LIMIT).contains(&maximum_m),
            "INT8 tiny-M workspace requires maximum M=1..={INT8_TINY_M_LIMIT}"
        );
        ensure!(k > 0 && k.is_multiple_of(4), "INT8 tiny-M workspace requires K divisible by 4");
        Ok(Self {
            quantized_input: runtime.alloc_uninit::<i8>(Shape::new([maximum_m, k]))?,
            input_scales: runtime.alloc_uninit::<f32>(Shape::new([maximum_m]))?,
            maximum_m,
            k,
        })
    }
}

pub(crate) fn quantize_weight_s8_per_channel(
    runtime: &CudaRuntime,
    weight: &Tensor<bf16>,
) -> Result<Int8PerChannelWeight> {
    ensure!(weight.rank() == 2, "INT8 weight must have rank 2");
    let n = weight.dims()[0];
    let k = weight.dims()[1];
    ensure!(n > 0 && k > 0, "INT8 weight cannot be empty");
    ensure!(k.is_multiple_of(4), "INT8 weight K must be divisible by 4");

    let mut data = runtime.alloc_uninit::<i8>(weight.shape().clone())?;
    let mut scales = runtime.alloc_uninit::<f32>(Shape::new([n]))?;
    unsafe {
        runtime.kernels().int8_tiny_m().launch_quantize_rows(
            runtime.stream(),
            QuantizeS8RowsLaunch {
                input: weight.storage(),
                output: data.storage_mut(),
                scales: scales.storage_mut(),
                rows: n,
                cols: k,
            },
        )?;
    }
    Ok(Int8PerChannelWeight { data, scales })
}

pub(crate) fn quantize_int8_tiny_m_input_into(
    runtime: &CudaRuntime,
    input: &Tensor<bf16>,
    workspace: &mut Int8TinyMWorkspace,
) -> Result<()> {
    ensure!(input.rank() == 2, "INT8 tiny-M input must have rank 2");
    let m = input.dims()[0];
    let k = input.dims()[1];
    ensure!(
        (1..=INT8_TINY_M_LIMIT).contains(&m),
        "INT8 tiny-M linear supports M=1..={INT8_TINY_M_LIMIT}, got {m}"
    );
    ensure!(
        workspace.maximum_m >= m && workspace.k == k,
        "INT8 tiny-M workspace shape mismatch"
    );
    workspace
        .quantized_input
        .set_logical_shape(Shape::new([m, k]))
        .context("failed to resize INT8 input scratch")?;
    workspace
        .input_scales
        .set_logical_shape(Shape::new([m]))
        .context("failed to resize INT8 scale scratch")?;
    unsafe {
        runtime.kernels().int8_tiny_m().launch_quantize_rows(
            runtime.stream(),
            QuantizeS8RowsLaunch {
                input: input.storage(),
                output: workspace.quantized_input.storage_mut(),
                scales: workspace.input_scales.storage_mut(),
                rows: m,
                cols: k,
            },
        )?;
    }
    Ok(())
}

pub(crate) fn silu_mul_packed_bf16_to_int8_tiny_m_into(
    runtime: &CudaRuntime,
    packed: &Tensor<bf16>,
    workspace: &mut Int8TinyMWorkspace,
) -> Result<()> {
    ensure!(packed.rank() == 2, "INT8 fused SwiGLU input must have rank 2");
    let m = packed.dims()[0];
    let packed_width = packed.dims()[1];
    ensure!(
        (1..=INT8_TINY_M_LIMIT).contains(&m),
        "INT8 fused SwiGLU supports M=1..={INT8_TINY_M_LIMIT}, got {m}"
    );
    ensure!(
        packed_width > 0 && packed_width.is_multiple_of(2),
        "INT8 fused SwiGLU packed width must be positive and even"
    );
    let k = packed_width / 2;
    ensure!(
        workspace.maximum_m >= m && workspace.k == k,
        "INT8 fused SwiGLU workspace shape mismatch"
    );
    workspace
        .quantized_input
        .set_logical_shape(Shape::new([m, k]))
        .context("failed to resize fused INT8 input scratch")?;
    workspace
        .input_scales
        .set_logical_shape(Shape::new([m]))
        .context("failed to resize fused INT8 scale scratch")?;
    unsafe {
        runtime.kernels().silu_mul().launch_packed_bf16_to_s8_dynamic(
            runtime.stream(),
            packed.storage(),
            workspace.quantized_input.storage_mut(),
            workspace.input_scales.storage_mut(),
            m,
            k,
        )?;
    }
    Ok(())
}

pub(crate) fn linear_int8_tiny_m_prequantized_into(
    runtime: &CudaRuntime,
    m: usize,
    weight: &Int8PerChannelWeight,
    workspace: &Int8TinyMWorkspace,
    output: &mut Tensor<bf16>,
) -> Result<()> {
    ensure!(weight.data.rank() == 2, "INT8 tiny-M weight must have rank 2");
    let n = weight.data.dims()[0];
    let k = weight.data.dims()[1];
    ensure!(
        (1..=INT8_TINY_M_LIMIT).contains(&m) && m <= workspace.maximum_m,
        "INT8 tiny-M prequantized M is invalid"
    );
    ensure!(workspace.k == k, "INT8 tiny-M prequantized K mismatch");
    ensure!(weight.scales.numel() == n, "INT8 tiny-M weight scale count mismatch");
    output.set_logical_shape(Shape::new([m, n]))?;
    unsafe {
        runtime.kernels().int8_tiny_m().launch_linear(
            runtime.stream(),
            TinyMInt8LinearLaunch {
                input: workspace.quantized_input.storage(),
                input_scales: workspace.input_scales.storage(),
                weight: weight.data.storage(),
                weight_scales: weight.scales.storage(),
                output: output.storage_mut(),
                m,
                n,
                k,
            },
        )?;
    }
    Ok(())
}

pub(crate) fn linear_int8_tiny_m_into(
    runtime: &CudaRuntime,
    input: &Tensor<bf16>,
    weight: &Int8PerChannelWeight,
    workspace: &mut Int8TinyMWorkspace,
    output: &mut Tensor<bf16>,
) -> Result<()> {
    ensure!(input.rank() == 2, "INT8 tiny-M input must have rank 2");
    let m = input.dims()[0];
    ensure!(weight.data.dims()[1] == input.dims()[1], "INT8 tiny-M weight/input K mismatch");
    quantize_int8_tiny_m_input_into(runtime, input, workspace)?;
    linear_int8_tiny_m_prequantized_into(runtime, m, weight, workspace, output)
}

pub(crate) fn linear_int8_tiny_m(
    runtime: &CudaRuntime,
    input: &Tensor<bf16>,
    weight: &Int8PerChannelWeight,
    workspace: &mut Int8TinyMWorkspace,
) -> Result<Tensor<bf16>> {
    ensure!(input.rank() == 2, "INT8 tiny-M input must have rank 2");
    let mut output = runtime.alloc_bf16(Shape::new([input.dims()[0], weight.data.dims()[0]]))?;
    linear_int8_tiny_m_into(runtime, input, weight, workspace, &mut output)?;
    Ok(output)
}
