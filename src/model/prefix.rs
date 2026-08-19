use anyhow::{Context as _, Result, ensure};
use half::bf16;

use crate::{
    cuda::CudaRuntime,
    tensor::{Shape, Tensor},
};

/// Fixed GPU-resident snapshots of all recurrent convolution states.
///
/// A checkpoint slot is shared across convolution layers: layer `i` stores its
/// state for the same logical prefix in `layers[i][checkpoint_slot, ..]`.
/// Capture/restore are same-stream D2D copies, so no host synchronization is
/// introduced on prefix publication or reuse.
pub(crate) struct ConvCheckpointPool {
    layers: Vec<Tensor<bf16>>,
    free_slots: Vec<u32>,
    elements_per_state: usize,
}

impl ConvCheckpointPool {
    pub(crate) fn new(
        runtime: &CudaRuntime,
        convolution_layers: usize,
        capacity: usize,
        hidden_size: usize,
        state_width: usize,
    ) -> Result<Self> {
        ensure!(convolution_layers > 0, "checkpoint pool needs convolution layers");
        ensure!(capacity > 0, "checkpoint pool capacity must be positive");
        ensure!(hidden_size > 0 && state_width > 0, "invalid convolution state shape");
        ensure!(capacity <= u32::MAX as usize, "checkpoint capacity exceeds u32");
        let elements_per_state = hidden_size
            .checked_mul(state_width)
            .context("convolution checkpoint state size overflow")?;
        let mut layers = Vec::with_capacity(convolution_layers);
        for _ in 0..convolution_layers {
            layers.push(runtime.zeros::<bf16>(Shape::new([
                capacity,
                hidden_size,
                state_width,
            ]))?);
        }
        let free_slots = (0..capacity).rev().map(|slot| slot as u32).collect();
        Ok(Self {
            layers,
            free_slots,
            elements_per_state,
        })
    }

    pub(crate) fn acquire(&mut self) -> Option<u32> {
        self.free_slots.pop()
    }

    pub(crate) fn release(&mut self, slot: u32) {
        self.free_slots.push(slot);
    }

    pub(crate) fn capture_layer(
        &mut self,
        runtime: &CudaRuntime,
        checkpoint_slot: u32,
        convolution_index: usize,
        source: &Tensor<bf16>,
        request_slot: usize,
    ) -> Result<()> {
        let destination = self
            .layers
            .get_mut(convolution_index)
            .context("convolution checkpoint layer out of range")?;
        let source_start = request_slot
            .checked_mul(self.elements_per_state)
            .context("convolution request-state offset overflow")?;
        let destination_start = usize::try_from(checkpoint_slot)?
            .checked_mul(self.elements_per_state)
            .context("convolution checkpoint offset overflow")?;
        runtime.copy_bf16_range(
            source,
            source_start,
            destination,
            destination_start,
            self.elements_per_state,
        )
    }

    pub(crate) fn restore_layer(
        &self,
        runtime: &CudaRuntime,
        checkpoint_slot: u32,
        convolution_index: usize,
        destination: &mut Tensor<bf16>,
        request_slot: usize,
    ) -> Result<()> {
        let source = self
            .layers
            .get(convolution_index)
            .context("convolution checkpoint layer out of range")?;
        let source_start = usize::try_from(checkpoint_slot)?
            .checked_mul(self.elements_per_state)
            .context("convolution checkpoint offset overflow")?;
        let destination_start = request_slot
            .checked_mul(self.elements_per_state)
            .context("convolution request-state offset overflow")?;
        runtime.copy_bf16_range(
            source,
            source_start,
            destination,
            destination_start,
            self.elements_per_state,
        )
    }

    pub(crate) fn capacity(&self) -> usize {
        self.layers
            .first()
            .map(|layer| layer.dims()[0])
            .unwrap_or(0)
    }

    pub(crate) fn available(&self) -> usize {
        self.free_slots.len()
    }
}
