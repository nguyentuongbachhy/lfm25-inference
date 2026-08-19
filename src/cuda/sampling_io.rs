use anyhow::{Context as _, Result, ensure};
use cudarc::driver::PinnedHostSlice;

use crate::tensor::Tensor;

use super::CudaRuntime;

impl CudaRuntime {
    pub(crate) fn download_u32_prefix_into(
        &self,
        tensor: &Tensor<u32>,
        elements: usize,
        destination: &mut PinnedHostSlice<u32>,
    ) -> Result<()> {
        ensure!(
            elements <= tensor.storage_capacity(),
            "sampled token prefix exceeds device output capacity"
        );
        ensure!(
            elements <= destination.len(),
            "sampled token prefix exceeds pinned host capacity"
        );
        let logical = tensor
            .storage()
            .try_slice(0..elements)
            .context("invalid sampled token download range")?;
        let destination = destination
            .as_mut_slice()
            .context("failed to access pinned sampled-token output")?;
        self.stream()
            .memcpy_dtoh(&logical, &mut destination[..elements])
            .context("failed to download sampled token prefix")?;
        self.stream()
            .synchronize()
            .context("failed to synchronize sampled token download")
    }
}
