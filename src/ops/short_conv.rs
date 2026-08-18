use anyhow::{Result, ensure};
use half::bf16;

use crate::{
    cuda::{CudaRuntime, SegmentedShortConvLaunch, ShortConvLaunch},
    tensor::{Shape, Tensor},
};
#[cfg(test)]
use crate::cuda::RaggedShortConvLaunch;

pub fn short_conv_lfm2_bf16(
    runtime: &CudaRuntime,
    projected: &Tensor<bf16>,
    weight: &Tensor<bf16>,
    state: &mut Tensor<bf16>,
) -> Result<Tensor<bf16>> {
    ensure!(projected.rank() == 2, "short convolution projection must have rank 2");
    ensure!(weight.rank() == 3, "short convolution weight must have rank 3");
    let num_tokens = projected.dims()[0];
    let projected_width = projected.dims()[1];
    ensure!(projected_width.is_multiple_of(3), "short convolution projection width must be divisible by 3");
    let hidden_size = projected_width / 3;
    ensure!(weight.dims() == [hidden_size, 1, 3], "short convolution weight mismatch: expected [{hidden_size},1,3], got {:?}", weight.dims());
    ensure!(state.dims() == [hidden_size, 2], "short convolution state mismatch: expected [{hidden_size},2], got {:?}", state.dims());
    let mut output = runtime.alloc_bf16(Shape::new([num_tokens, hidden_size]))?;
    unsafe {
        runtime.kernels().short_conv().launch_lfm2_bf16(
            runtime.stream(),
            ShortConvLaunch {
                projected: projected.storage(),
                weight: weight.storage(),
                state: state.storage_mut(),
                output: output.storage_mut(),
                num_tokens,
                hidden_size,
            },
        )?;
    }
    Ok(output)
}

#[cfg(test)]
pub fn short_conv_ragged_lfm2_bf16(
    runtime: &CudaRuntime,
    projected: &Tensor<bf16>,
    weight: &Tensor<bf16>,
    states: &mut Tensor<bf16>,
    request_slots: &Tensor<u32>,
) -> Result<Tensor<bf16>> {
    ensure!(projected.rank() == 2, "ragged convolution projection must have rank 2");
    ensure!(states.rank() == 3, "ragged convolution states must have rank 3");
    let num_tokens = projected.dims()[0];
    let hidden_size = projected.dims()[1] / 3;
    ensure!(projected.dims()[1] == hidden_size * 3, "invalid projection width");
    ensure!(weight.dims() == [hidden_size, 1, 3], "convolution weight mismatch");
    ensure!(states.dims()[1..] == [hidden_size, 2], "convolution state mismatch");
    ensure!(request_slots.numel() == num_tokens, "request slot count mismatch");
    let num_request_slots = states.dims()[0];
    let mut output = runtime.alloc_bf16(Shape::new([num_tokens, hidden_size]))?;
    unsafe {
        runtime.kernels().short_conv().launch_ragged_lfm2_bf16(
            runtime.stream(),
            RaggedShortConvLaunch {
                projected: projected.storage(),
                weight: weight.storage(),
                states: states.storage_mut(),
                request_slots: request_slots.storage(),
                output: output.storage_mut(),
                num_tokens,
                hidden_size,
                num_request_slots,
            },
        )?;
    }
    Ok(output)
}

pub fn short_conv_segmented_lfm2_bf16(
    runtime: &CudaRuntime,
    projected: &Tensor<bf16>,
    weight: &Tensor<bf16>,
    states: &mut Tensor<bf16>,
    segment_offsets: &Tensor<u32>,
    segment_slots: &Tensor<u32>,
) -> Result<Tensor<bf16>> {
    ensure!(projected.rank() == 2 && states.rank() == 3, "invalid segmented convolution rank");
    let num_tokens = projected.dims()[0];
    let hidden_size = projected.dims()[1] / 3;
    let num_segments = segment_slots.numel();
    let num_request_slots = states.dims()[0];
    ensure!(num_segments > 0 && segment_offsets.numel() == num_segments + 1, "invalid segment metadata");
    ensure!(weight.dims() == [hidden_size, 1, 3], "segmented convolution weight mismatch");
    ensure!(states.dims()[1..] == [hidden_size, 2], "segmented convolution state mismatch");
    let mut output = runtime.alloc_bf16(Shape::new([num_tokens, hidden_size]))?;
    unsafe {
        runtime.kernels().short_conv().launch_segmented_lfm2_bf16(
            runtime.stream(),
            SegmentedShortConvLaunch {
                projected: projected.storage(),
                weight: weight.storage(),
                states: states.storage_mut(),
                segment_offsets: segment_offsets.storage(),
                segment_slots: segment_slots.storage(),
                output: output.storage_mut(),
                num_tokens,
                num_segments,
                hidden_size,
                num_request_slots,
            },
        )?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use crate::cuda::testing::{assert_close_bf16, readback};
    use super::*;

    #[test]
    fn short_conv_matches_causal_reference_and_updates_state() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let projected_host = [
            1.0, 2.0, 0.5, 1.0, 3.0, 4.0, 2.0, 1.0, 1.0, 0.5, 5.0, 6.0, 1.0, 3.0, 2.0, 1.0, 7.0, 8.0,
        ].map(bf16::from_f32);
        let weights_host = [0.25, 0.5, 1.0, -0.5, 0.25, 2.0].map(bf16::from_f32);
        let projected = runtime.upload(&projected_host, Shape::new([3, 6]))?;
        let weights = runtime.upload(&weights_host, Shape::new([2, 1, 3]))?;
        let mut state = runtime.zeros::<bf16>(Shape::new([2, 2]))?;
        let output = short_conv_lfm2_bf16(&runtime, &projected, &weights, &mut state)?;
        let actual = readback(&runtime, &output)?;
        let state_actual = readback(&runtime, &state)?;
        let gated = [[3.0, 8.0], [10.0, 6.0], [7.0, 24.0]];
        let c = [[0.5, 1.0], [1.0, 0.5], [2.0, 1.0]];
        let weights = [[0.25, 0.5, 1.0], [-0.5, 0.25, 2.0]];
        let mut history = [[0.0f32; 2]; 2];
        let mut expected = Vec::new();
        for token in 0..3 {
            for channel in 0..2 {
                let convolution = weights[channel][0] * history[channel][0]
                    + weights[channel][1] * history[channel][1]
                    + weights[channel][2] * gated[token][channel];
                expected.push(bf16::from_f32(c[token][channel] * convolution));
                history[channel] = [history[channel][1], gated[token][channel]];
            }
        }
        let expected_state = [10.0, 7.0, 6.0, 24.0].map(bf16::from_f32);
        assert_close_bf16(&actual, &expected, 0.1, 0.01);
        assert_close_bf16(&state_actual, &expected_state, 0.0, 0.0);
        Ok(())
    }

    #[test]
    fn ragged_short_conv_updates_independent_slot_states() -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let projected_host = [2.0, 3.0, 1.0, 1.0, 4.0, 5.0, 7.0, 11.0, 2.0, 3.0, 13.0, 17.0].map(bf16::from_f32);
        let projected = runtime.upload(&projected_host, Shape::new([2, 6]))?;
        let weights = runtime.upload(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0].map(bf16::from_f32), Shape::new([2, 1, 3]))?;
        let mut states = runtime.zeros::<bf16>(Shape::new([2, 2, 2]))?;
        let slots = runtime.upload(&[0u32, 1], Shape::new([2]))?;
        let output = short_conv_ragged_lfm2_bf16(&runtime, &projected, &weights, &mut states, &slots)?;
        let actual = readback(&runtime, &output)?;
        let expected = [8.0, 15.0, 182.0, 561.0].map(bf16::from_f32);
        assert_close_bf16(&actual, &expected, 0.0, 0.0);
        let state = readback(&runtime, &states)?;
        let expected_state = [0.0, 8.0, 0.0, 15.0, 0.0, 91.0, 0.0, 187.0].map(bf16::from_f32);
        assert_close_bf16(&state, &expected_state, 0.0, 0.0);
        Ok(())
    }
}
