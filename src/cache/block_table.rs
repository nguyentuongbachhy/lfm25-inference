use anyhow::{Result, ensure};

pub struct FixedBlockTables {
    entries: Vec<u32>,
    slots: usize,
    pages_per_slot: usize,
}

impl FixedBlockTables {
    pub fn new(slots: usize, pages_per_slot: usize) -> Result<Self> {
        ensure!(slots > 0, "block tables require request slots");
        ensure!(pages_per_slot > 0, "block tables require logical pages");
        let elements = slots
            .checked_mul(pages_per_slot)
            .ok_or_else(|| anyhow::anyhow!("block table size overflow"))?;
        Ok(Self {
            entries: vec![u32::MAX; elements],
            slots,
            pages_per_slot,
        })
    }

    #[cfg(test)]
    pub fn slot(&self, slot: usize) -> Result<&[u32]> {
        ensure!(slot < self.slots, "block table slot out of range");
        let start = slot * self.pages_per_slot;
        Ok(&self.entries[start..start + self.pages_per_slot])
    }

    pub fn slot_mut(&mut self, slot: usize) -> Result<&mut [u32]> {
        ensure!(slot < self.slots, "block table slot out of range");
        let start = slot * self.pages_per_slot;
        Ok(&mut self.entries[start..start + self.pages_per_slot])
    }

    #[cfg(test)]
    pub fn as_slice(&self) -> &[u32] {
        &self.entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_have_stable_disjoint_slot_ranges() -> Result<()> {
        let mut tables = FixedBlockTables::new(3, 4)?;
        tables.slot_mut(1)?[2] = 7;
        assert_eq!(tables.slot(1)?[2], 7);
        assert!(tables.slot(0)?.iter().all(|page| *page == u32::MAX));
        assert_eq!(tables.as_slice().len(), 12);
        Ok(())
    }
}
