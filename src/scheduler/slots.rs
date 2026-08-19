use anyhow::{Result, ensure};

use super::{RequestPhase, SequenceRequest};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RequestSlotId(pub u32);

pub struct RequestSlots {
    entries: Vec<SequenceRequest>,
    free: Vec<u32>,
}

impl RequestSlots {
    pub fn new(capacity: usize, maximum_sequence_tokens: usize) -> Result<Self> {
        ensure!(capacity > 0, "request slot capacity must be positive");
        ensure!(
            capacity <= u32::MAX as usize,
            "request slot capacity exceeds u32"
        );
        let mut entries = Vec::with_capacity(capacity);
        let mut free = Vec::with_capacity(capacity);
        for slot in 0..capacity {
            entries.push(SequenceRequest::vacant(maximum_sequence_tokens));
            free.push((capacity - slot - 1) as u32);
        }
        Ok(Self { entries, free })
    }

    pub fn acquire(&mut self) -> Option<RequestSlotId> {
        self.free.pop().map(RequestSlotId)
    }

    pub fn release(&mut self, slot: RequestSlotId) -> Result<()> {
        let index = slot.0 as usize;
        ensure!(index < self.entries.len(), "request slot out of range");
        ensure!(
            self.entries[index].phase != RequestPhase::Free,
            "request slot already free"
        );
        self.entries[index].clear();
        self.free.push(slot.0);
        Ok(())
    }

    pub fn get(&self, slot: RequestSlotId) -> Result<&SequenceRequest> {
        self.entries
            .get(slot.0 as usize)
            .ok_or_else(|| anyhow::anyhow!("request slot out of range"))
    }

    pub fn get_mut(&mut self, slot: RequestSlotId) -> Result<&mut SequenceRequest> {
        self.entries
            .get_mut(slot.0 as usize)
            .ok_or_else(|| anyhow::anyhow!("request slot out of range"))
    }

    pub fn entries(&self) -> &[SequenceRequest] {
        &self.entries
    }

    pub fn free_count(&self) -> usize {
        self.free.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::RequestInit;

    #[test]
    fn slots_reuse_preallocated_token_capacity() -> Result<()> {
        let mut slots = RequestSlots::new(2, 32)?;
        let slot = slots.acquire().expect("available slot");
        let capacity = slots.get(slot)?.tokens().len();
        slots
            .get_mut(slot)?
            .initialize(RequestInit::new(1, &[1, 2], 16, 0, 400_000, 50_000, 1))?;
        assert_eq!(slots.get(slot)?.tokens(), &[1, 2]);
        slots.release(slot)?;
        let reused = slots.acquire().expect("reused slot");
        assert_eq!(reused, slot);
        assert_eq!(slots.get(reused)?.tokens().len(), capacity);
        Ok(())
    }
}
