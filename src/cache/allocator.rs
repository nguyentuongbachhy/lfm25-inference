use std::fmt;

use anyhow::{Result, ensure};

use super::KvPageSize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvAllocationError {
    PhysicalPagesExhausted,
    LogicalReservationExhausted,
    SequenceCapacityExceeded,
}

impl fmt::Display for KvAllocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::PhysicalPagesExhausted => "physical KV pages exhausted",
            Self::LogicalReservationExhausted => "logical KV reservation exhausted",
            Self::SequenceCapacityExceeded => "sequence block table capacity exceeded",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for KvAllocationError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvPoolSnapshot {
    pub total_pages: usize,
    pub free_pages: usize,
    pub allocated_pages: usize,
    pub reserved_pages: usize,
    pub peak_allocated_pages: usize,
}

pub struct KvPageAllocator {
    page_size: KvPageSize,
    free_pages: Vec<u32>,
    total_pages: usize,
    reserved_pages: usize,
    peak_allocated_pages: usize,
}

impl KvPageAllocator {
    pub fn new(total_pages: usize, page_size: KvPageSize) -> Result<Self> {
        ensure!(total_pages > 0, "KV page allocator requires physical pages");
        ensure!(
            total_pages <= u32::MAX as usize,
            "KV physical page count exceeds u32"
        );
        let mut free_pages = Vec::with_capacity(total_pages);
        for page in (0..total_pages).rev() {
            free_pages.push(page as u32);
        }
        Ok(Self {
            page_size,
            free_pages,
            total_pages,
            reserved_pages: 0,
            peak_allocated_pages: 0,
        })
    }

    pub fn try_reserve_tokens(
        &mut self,
        maximum_tokens: usize,
    ) -> Result<usize, KvAllocationError> {
        if maximum_tokens == 0 {
            return Err(KvAllocationError::SequenceCapacityExceeded);
        }
        let pages = maximum_tokens.div_ceil(self.page_size.value());
        let next = self
            .reserved_pages
            .checked_add(pages)
            .ok_or(KvAllocationError::LogicalReservationExhausted)?;
        if next > self.total_pages {
            return Err(KvAllocationError::LogicalReservationExhausted);
        }
        self.reserved_pages = next;
        Ok(pages)
    }

    pub fn release_reservation(&mut self, pages: usize) {
        self.reserved_pages = self.reserved_pages.saturating_sub(pages);
    }

    pub fn grow_sequence(
        &mut self,
        current_tokens: usize,
        target_tokens: usize,
        block_table: &mut [u32],
    ) -> Result<usize, KvAllocationError> {
        if target_tokens < current_tokens {
            return Err(KvAllocationError::SequenceCapacityExceeded);
        }
        let current_pages = current_tokens.div_ceil(self.page_size.value());
        let target_pages = target_tokens.div_ceil(self.page_size.value());
        if target_pages > block_table.len() {
            return Err(KvAllocationError::SequenceCapacityExceeded);
        }
        let needed = target_pages.saturating_sub(current_pages);
        if needed > self.free_pages.len() {
            return Err(KvAllocationError::PhysicalPagesExhausted);
        }
        for logical_page in current_pages..target_pages {
            let physical_page = self
                .free_pages
                .pop()
                .ok_or(KvAllocationError::PhysicalPagesExhausted)?;
            block_table[logical_page] = physical_page;
        }
        self.peak_allocated_pages = self
            .peak_allocated_pages
            .max(self.total_pages - self.free_pages.len());
        Ok(needed)
    }

    pub fn release_sequence(&mut self, tokens: usize, block_table: &mut [u32]) {
        let pages = tokens
            .div_ceil(self.page_size.value())
            .min(block_table.len());
        for entry in &mut block_table[..pages] {
            if *entry != u32::MAX {
                self.free_pages.push(*entry);
                *entry = u32::MAX;
            }
        }
    }

    pub fn snapshot(&self) -> KvPoolSnapshot {
        KvPoolSnapshot {
            total_pages: self.total_pages,
            free_pages: self.free_pages.len(),
            allocated_pages: self.total_pages - self.free_pages.len(),
            reserved_pages: self.reserved_pages,
            peak_allocated_pages: self.peak_allocated_pages,
        }
    }

    pub(crate) fn reset_peak(&mut self) {
        self.peak_allocated_pages = self.total_pages - self.free_pages.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_grows_incrementally_and_reclaims_pages() -> Result<()> {
        let mut allocator = KvPageAllocator::new(8, KvPageSize::P16)?;
        let reserved = allocator.try_reserve_tokens(64)?;
        assert_eq!(reserved, 4);
        let mut table = [u32::MAX; 4];
        assert_eq!(allocator.grow_sequence(0, 17, &mut table)?, 2);
        assert_eq!(allocator.snapshot().allocated_pages, 2);
        assert_eq!(allocator.grow_sequence(17, 48, &mut table)?, 1);
        allocator.release_sequence(48, &mut table);
        allocator.release_reservation(reserved);
        assert_eq!(allocator.snapshot().free_pages, 8);
        assert!(table.iter().all(|page| *page == u32::MAX));
        Ok(())
    }

    #[test]
    fn reservation_prevents_collective_growth_oom() -> Result<()> {
        let mut allocator = KvPageAllocator::new(4, KvPageSize::P16)?;
        assert_eq!(allocator.try_reserve_tokens(48)?, 3);
        assert_eq!(
            allocator.try_reserve_tokens(32),
            Err(KvAllocationError::LogicalReservationExhausted)
        );
        Ok(())
    }
}
