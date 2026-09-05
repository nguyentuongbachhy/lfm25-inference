use std::sync::Arc;

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, PushKernelArg};
use half::bf16;

use crate::cuda::{launch::KernelLaunch, module::load_function};

use super::kernel_set::KernelSet;

const MAX_BLOCK_SIZE: u32 = 256;

pub(crate) struct ShortConvLaunch<'a> {
    pub(crate) projected: &'a CudaSlice<bf16>,
    pub(crate) weight: &'a CudaSlice<bf16>,
    pub(crate) state: &'a mut CudaSlice<bf16>,
    pub(crate) output: &'a mut CudaSlice<bf16>,
    pub(crate) num_tokens: usize,
    pub(crate) hidden_size: usize,
}

pub(crate) struct ShortConvWithHistoryLaunch<'a> {
    pub(crate) projected: &'a CudaSlice<bf16>,
    pub(crate) weight: &'a CudaSlice<bf16>,
    pub(crate) state: &'a mut CudaSlice<bf16>,
    pub(crate) output: &'a mut CudaSlice<bf16>,
    pub(crate) state_history: &'a mut CudaSlice<bf16>,
    pub(crate) num_tokens: usize,
    pub(crate) hidden_size: usize,
}

#[cfg(test)]
pub(crate) struct RaggedShortConvLaunch<'a> {
    pub(crate) projected: &'a CudaSlice<bf16>,
    pub(crate) weight: &'a CudaSlice<bf16>,
    pub(crate) states: &'a mut CudaSlice<bf16>,
    pub(crate) request_slots: &'a CudaSlice<u32>,
    pub(crate) output: &'a mut CudaSlice<bf16>,
    pub(crate) num_tokens: usize,
    pub(crate) hidden_size: usize,
    pub(crate) num_request_slots: usize,
}

pub(crate) struct SegmentedShortConvLaunch<'a> {
    pub(crate) projected: &'a CudaSlice<bf16>,
    pub(crate) weight: &'a CudaSlice<bf16>,
    pub(crate) states: &'a mut CudaSlice<bf16>,
    pub(crate) segment_offsets: &'a CudaSlice<u32>,
    pub(crate) segment_slots: &'a CudaSlice<u32>,
    pub(crate) output: &'a mut CudaSlice<bf16>,
    pub(crate) num_tokens: usize,
    pub(crate) num_segments: usize,
    pub(crate) hidden_size: usize,
    pub(crate) num_request_slots: usize,
}

pub(crate) struct ShortConvKernels {
    lfm2: KernelLaunch,
    lfm2_with_history: KernelLaunch,
    #[cfg(test)]
    ragged_lfm2: KernelLaunch,
    segmented_lfm2: KernelLaunch,
}

impl KernelSet for ShortConvKernels {
    const MODULE_NAME: &'static str = "short_conv";
    const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/short_conv.ptx"));

    fn from_module(module: Arc<CudaModule>) -> Result<Self> {
        let function = load_function(&module, Self::MODULE_NAME, "short_conv_lfm2_bf16")?;
        let with_history_function = load_function(
            &module,
            Self::MODULE_NAME,
            "short_conv_lfm2_bf16_with_history",
        )?;
        #[cfg(test)]
        let ragged_function =
            load_function(&module, Self::MODULE_NAME, "short_conv_ragged_lfm2_bf16")?;
        let segmented_function =
            load_function(&module, Self::MODULE_NAME, "short_conv_segmented_lfm2_bf16")?;
        Ok(Self {
            lfm2: KernelLaunch::new(function, MAX_BLOCK_SIZE)?,
            lfm2_with_history: KernelLaunch::new(with_history_function, MAX_BLOCK_SIZE)?,
            #[cfg(test)]
            ragged_lfm2: KernelLaunch::new(ragged_function, MAX_BLOCK_SIZE)?,
            segmented_lfm2: KernelLaunch::new(segmented_function, MAX_BLOCK_SIZE)?,
        })
    }
}

impl ShortConvKernels {
    pub(crate) unsafe fn launch_segmented_lfm2_bf16(
        &self,
        stream: &CudaStream,
        launch: SegmentedShortConvLaunch<'_>,
    ) -> Result<()> {
        let SegmentedShortConvLaunch {
            projected,
            weight,
            states,
            segment_offsets,
            segment_slots,
            output,
            num_tokens,
            num_segments,
            hidden_size,
            num_request_slots,
        } = launch;
        ensure!(
            num_tokens > 0 && num_segments > 0 && hidden_size > 0,
            "invalid segmented convolution shape"
        );
        ensure!(
            segment_offsets.len() > num_segments,
            "segment offsets too small"
        );
        ensure!(
            segment_slots.len() >= num_segments,
            "segment slots too small"
        );
        let projected_required = num_tokens
            .checked_mul(hidden_size)
            .and_then(|v| v.checked_mul(3))
            .context("segmented projection overflow")?;
        let output_required = num_tokens
            .checked_mul(hidden_size)
            .context("segmented output overflow")?;
        let state_required = num_request_slots
            .checked_mul(hidden_size)
            .and_then(|v| v.checked_mul(2))
            .context("segmented state overflow")?;
        ensure!(
            projected.len() >= projected_required && output.len() >= output_required,
            "segmented tensor storage too small"
        );
        ensure!(
            states.len() >= state_required && weight.len() >= hidden_size * 3,
            "segmented state/weight too small"
        );
        let work = num_segments
            .checked_mul(hidden_size)
            .context("segmented work overflow")?;
        let config = self.segmented_lfm2.policy().for_work_items(work)?;
        let mut args = stream.launch_builder(self.segmented_lfm2.function());
        args.arg(projected)
            .arg(weight)
            .arg(states)
            .arg(segment_offsets)
            .arg(segment_slots)
            .arg(output)
            .arg(&num_segments)
            .arg(&hidden_size)
            .arg(&num_request_slots);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) unsafe fn launch_ragged_lfm2_bf16(
        &self,
        stream: &CudaStream,
        launch: RaggedShortConvLaunch<'_>,
    ) -> Result<()> {
        let RaggedShortConvLaunch {
            projected,
            weight,
            states,
            request_slots,
            output,
            num_tokens,
            hidden_size,
            num_request_slots,
        } = launch;
        ensure!(
            num_tokens > 0 && hidden_size > 0,
            "invalid ragged convolution shape"
        );
        ensure!(request_slots.len() >= num_tokens, "request slots too small");
        let projected_required = num_tokens
            .checked_mul(hidden_size)
            .and_then(|value| value.checked_mul(3))
            .context("ragged convolution projection overflow")?;
        let output_required = num_tokens
            .checked_mul(hidden_size)
            .context("ragged convolution output overflow")?;
        let state_required = num_request_slots
            .checked_mul(hidden_size)
            .and_then(|value| value.checked_mul(2))
            .context("ragged convolution state overflow")?;
        ensure!(
            projected.len() >= projected_required,
            "projection storage too small"
        );
        ensure!(weight.len() >= hidden_size * 3, "weight storage too small");
        ensure!(states.len() >= state_required, "state storage too small");
        ensure!(output.len() >= output_required, "output storage too small");
        let config = self.ragged_lfm2.policy().for_work_items(output_required)?;
        let mut args = stream.launch_builder(self.ragged_lfm2.function());
        args.arg(projected)
            .arg(weight)
            .arg(states)
            .arg(request_slots)
            .arg(output)
            .arg(&num_tokens)
            .arg(&hidden_size)
            .arg(&num_request_slots);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }

    pub(crate) unsafe fn launch_lfm2_bf16(
        &self,
        stream: &CudaStream,
        launch: ShortConvLaunch<'_>,
    ) -> Result<()> {
        let ShortConvLaunch {
            projected,
            weight,
            state,
            output,
            num_tokens,
            hidden_size,
        } = launch;
        ensure!(
            num_tokens > 0,
            "short convolution requires at least one token"
        );
        ensure!(
            hidden_size > 0,
            "short convolution hidden size must be positive"
        );
        let projected_required = num_tokens
            .checked_mul(hidden_size)
            .and_then(|value| value.checked_mul(3))
            .context("short convolution projection size overflow")?;
        let output_required = num_tokens
            .checked_mul(hidden_size)
            .context("short convolution output size overflow")?;
        ensure!(
            projected.len() >= projected_required,
            "short convolution projection storage too small"
        );
        ensure!(
            weight.len() >= hidden_size * 3,
            "short convolution weight storage too small"
        );
        ensure!(
            state.len() >= hidden_size * 2,
            "short convolution state storage too small"
        );
        ensure!(
            output.len() >= output_required,
            "short convolution output storage too small"
        );
        let config = self.lfm2.policy().for_work_items(hidden_size)?;
        let mut args = stream.launch_builder(self.lfm2.function());
        args.arg(projected)
            .arg(weight)
            .arg(state)
            .arg(output)
            .arg(&num_tokens)
            .arg(&hidden_size);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }

    pub(crate) unsafe fn launch_lfm2_bf16_with_history(
        &self,
        stream: &CudaStream,
        launch: ShortConvWithHistoryLaunch<'_>,
    ) -> Result<()> {
        let ShortConvWithHistoryLaunch {
            projected,
            weight,
            state,
            output,
            state_history,
            num_tokens,
            hidden_size,
        } = launch;
        ensure!(
            num_tokens > 0,
            "short convolution requires at least one token"
        );
        ensure!(
            hidden_size > 0,
            "short convolution hidden size must be positive"
        );
        let projected_required = num_tokens
            .checked_mul(hidden_size)
            .and_then(|value| value.checked_mul(3))
            .context("short convolution projection size overflow")?;
        let output_required = num_tokens
            .checked_mul(hidden_size)
            .context("short convolution output size overflow")?;
        let history_required = num_tokens
            .checked_mul(hidden_size)
            .and_then(|value| value.checked_mul(2))
            .context("short convolution state history size overflow")?;
        ensure!(
            projected.len() >= projected_required,
            "short convolution projection storage too small"
        );
        ensure!(
            weight.len() >= hidden_size * 3,
            "short convolution weight storage too small"
        );
        ensure!(
            state.len() >= hidden_size * 2,
            "short convolution state storage too small"
        );
        ensure!(
            output.len() >= output_required,
            "short convolution output storage too small"
        );
        ensure!(
            state_history.len() >= history_required,
            "short convolution state history storage too small"
        );
        let config = self
            .lfm2_with_history
            .policy()
            .for_work_items(hidden_size)?;
        let mut args = stream.launch_builder(self.lfm2_with_history.function());
        args.arg(projected)
            .arg(weight)
            .arg(state)
            .arg(output)
            .arg(state_history)
            .arg(&num_tokens)
            .arg(&hidden_size);
        unsafe {
            args.launch(config)?;
        }
        Ok(())
    }
}
