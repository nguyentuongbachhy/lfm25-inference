use anyhow::Result;

use super::Engine;

impl Engine {
    /// Permanently transitions CUDA runtime state into single-owner serving mode.
    ///
    /// Call this only after the Engine has reached the thread that will own all
    /// subsequent GPU work. The transition removes synchronization from pooled
    /// temporary storage and cuBLASLt plan lookup on that owner thread.
    pub(crate) fn enter_serving_owner_mode(&self) -> Result<()> {
        self.runtime.enter_serving_owner_mode()
    }
}
