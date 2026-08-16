use std::ffi::c_int;

use anyhow::{Context as _, Result, ensure};
use cudarc::driver::{CudaFunction, LaunchConfig};

extern "C" fn no_dynamic_smem(_block_size: c_int) -> usize {
    0
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LaunchPolicy {
    block_size: u32,
}

impl LaunchPolicy {
    pub(crate) fn from_function(function: &CudaFunction, block_size_limit: u32) -> Result<Self> {
        Self::from_function_with_multiple(function, block_size_limit, 1)
    }

    pub(crate) fn from_function_with_multiple(
        function: &CudaFunction,
        block_size_limit: u32,
        block_multiple: u32,
    ) -> Result<Self> {
        ensure!(
            block_multiple > 0,
            "block multiple must be greater than zero",
        );

        let kernel_max_threads = function
            .max_threads_per_block()
            .context("failed to query kernel max threads per block")?;

        ensure!(
            kernel_max_threads > 0,
            "kernel reports invalid max threads per block: {kernel_max_threads}",
        );

        let kernel_max_threads = kernel_max_threads as u32;

        let effective_limit = if block_size_limit == 0 {
            kernel_max_threads
        } else {
            block_size_limit.min(kernel_max_threads)
        };

        let (_min_grid_size, suggested_block_size) = function
            .occupancy_max_potential_block_size(no_dynamic_smem, 0, effective_limit, None)
            .context("failed to query kernel occupancy")?;

        ensure!(
            suggested_block_size > 0,
            "occupancy API returned block_size=0",
        );

        let block_size = if block_multiple == 1 {
            suggested_block_size
        } else {
            suggested_block_size / block_multiple * block_multiple
        };

        ensure!(
            block_size >= block_multiple,
            "cannot satisfy block multiple {block_multiple} \
             with occupancy block size {suggested_block_size}",
        );

        ensure!(
            block_size <= effective_limit,
            "block size {block_size} exceeds kernel limit {effective_limit}",
        );

        Ok(Self { block_size })
    }

    pub(crate) fn block_size(&self) -> u32 {
        self.block_size
    }

    pub(crate) fn for_work_items(&self, work_items: usize) -> Result<LaunchConfig> {
        let block_size = self.block_size as usize;

        let grid_size = work_items.max(1).div_ceil(block_size);

        let grid_x = u32::try_from(grid_size).context("CUDA grid size exceeds u32")?;

        Ok(LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (self.block_size, 1, 1),
            shared_mem_bytes: 0,
        })
    }

    pub(crate) fn exact_blocks(&self, blocks: usize) -> Result<LaunchConfig> {
        ensure!(blocks > 0, "CUDA launch requires at least one block",);

        let grid_x = u32::try_from(blocks).context("CUDA grid size exceeds u32")?;

        Ok(LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (self.block_size, 1, 1),
            shared_mem_bytes: 0,
        })
    }
}

pub(crate) struct KernelLaunch {
    function: CudaFunction,
    policy: LaunchPolicy,
}

impl KernelLaunch {
    pub(crate) fn new(function: CudaFunction, block_size_limit: u32) -> Result<Self> {
        let policy = LaunchPolicy::from_function(&function, block_size_limit)?;

        Ok(Self { function, policy })
    }

    pub(crate) fn new_with_multiple(
        function: CudaFunction,
        block_size_limit: u32,
        block_multiple: u32,
    ) -> Result<Self> {
        let policy =
            LaunchPolicy::from_function_with_multiple(&function, block_size_limit, block_multiple)?;

        Ok(Self { function, policy })
    }

    pub(crate) fn function(&self) -> &CudaFunction {
        &self.function
    }

    pub(crate) fn policy(&self) -> LaunchPolicy {
        self.policy
    }
}
