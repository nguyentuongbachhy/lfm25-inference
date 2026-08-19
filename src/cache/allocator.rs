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
    pub cached_pages: usize,
    pub peak_allocated_pages: usize,
}

/// Physical KV page allocator with request references plus long-lived radix
/// cache pins. All mutations happen on the single GPU-owner thread, so the
/// reference counts deliberately remain non-atomic.
pub struct KvPageAllocator {
    page_size: KvPageSize,
    free_pages: Vec<u32>,
    page_refs: Vec<u32>,
    total_pages: usize,
    reserved_pages: usize,
    cached_pages: usize,
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
            page_refs: vec![0; total_pages],
            total_pages,
            reserved_pages: 0,
            cached_pages: 0,
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
        self.try_reserve_pages(maximum_tokens.div_ceil(self.page_size.value()))
    }

    /// Reserves only pages that may need private allocation. Shared immutable
    /// prefix pages are charged once through `cached_pages`, not once/request.
    pub fn try_reserve_pages(&mut self, pages: usize) -> Result<usize, KvAllocationError> {
        let next = self
            .reserved_pages
            .checked_add(pages)
            .ok_or(KvAllocationError::LogicalReservationExhausted)?;
        let committed = next
            .checked_add(self.cached_pages)
            .ok_or(KvAllocationError::LogicalReservationExhausted)?;
        if committed > self.total_pages {
            return Err(KvAllocationError::LogicalReservationExhausted);
        }
        self.reserved_pages = next;
        Ok(pages)
    }

    pub fn release_reservation(&mut self, pages: usize) {
        self.reserved_pages = self.reserved_pages.saturating_sub(pages);
    }

    pub fn pin_cached_pages(&mut self, pages: &[u32]) -> Result<(), KvAllocationError> {
        if self.cached_pages.saturating_add(pages.len()) > self.total_pages {
            return Err(KvAllocationError::PhysicalPagesExhausted);
        }
        for &page in pages {
            let reference = self.reference_mut(page)?;
            if *reference == 0 {
                return Err(KvAllocationError::PhysicalPagesExhausted);
            }
            *reference = reference
                .checked_add(1)
                .ok_or(KvAllocationError::PhysicalPagesExhausted)?;
        }
        self.cached_pages += pages.len();
        Ok(())
    }

    pub fn retain_pages(&mut self, pages: &[u32]) -> Result<(), KvAllocationError> {
        for &page in pages {
            let reference = self.reference_mut(page)?;
            if *reference == 0 {
                return Err(KvAllocationError::PhysicalPagesExhausted);
            }
            *reference = reference
                .checked_add(1)
                .ok_or(KvAllocationError::PhysicalPagesExhausted)?;
        }
        Ok(())
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
        for entry in block_table
            .iter_mut()
            .take(target_pages)
            .skip(current_pages)
        {
            let page = self
                .free_pages
                .pop()
                .ok_or(KvAllocationError::PhysicalPagesExhausted)?;
            let reference = self.reference_mut(page)?;
            if *reference != 0 {
                return Err(KvAllocationError::PhysicalPagesExhausted);
            }
            *reference = 1;
            *entry = page;
        }
        self.peak_allocated_pages = self
            .peak_allocated_pages
            .max(self.total_pages - self.free_pages.len());
        Ok(needed)
    }

    /// Keeps the historical infallible release API used by the model cache.
    /// Invalid page ownership is an internal invariant violation; in release
    /// builds we avoid panicking the serving process and simply leave the page
    /// unavailable rather than risking a double-free into `free_pages`.
    pub fn release_sequence(&mut self, tokens: usize, block_table: &mut [u32]) {
        let pages = tokens
            .div_ceil(self.page_size.value())
            .min(block_table.len());
        for entry in &mut block_table[..pages] {
            if *entry != u32::MAX {
                self.release_page(*entry);
                *entry = u32::MAX;
            }
        }
    }

    fn reference_mut(&mut self, page: u32) -> Result<&mut u32, KvAllocationError> {
        self.page_refs
            .get_mut(page as usize)
            .ok_or(KvAllocationError::PhysicalPagesExhausted)
    }

    fn release_page(&mut self, page: u32) {
        let Some(reference) = self.page_refs.get_mut(page as usize) else {
            debug_assert!(false, "KV page out of range during release");
            return;
        };
        if *reference == 0 {
            debug_assert!(false, "KV page double release");
            return;
        }
        *reference -= 1;
        if *reference == 0 {
            self.free_pages.push(page);
        }
    }

    pub fn snapshot(&self) -> KvPoolSnapshot {
        KvPoolSnapshot {
            total_pages: self.total_pages,
            free_pages: self.free_pages.len(),
            allocated_pages: self.total_pages - self.free_pages.len(),
            reserved_pages: self.reserved_pages,
            cached_pages: self.cached_pages,
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

    #[test]
    fn cached_page_survives_request_release_and_can_be_shared() -> Result<()> {
        let mut allocator = KvPageAllocator::new(4, KvPageSize::P16)?;
        let reserved = allocator.try_reserve_tokens(16)?;
        let mut first = [u32::MAX; 1];
        allocator.grow_sequence(0, 16, &mut first)?;
        let page = first[0];
        allocator.pin_cached_pages(&[page])?;
        allocator.release_sequence(16, &mut first);
        allocator.release_reservation(reserved);
        assert_eq!(allocator.snapshot().allocated_pages, 1);
        assert_eq!(allocator.snapshot().cached_pages, 1);

        allocator.retain_pages(&[page])?;
        let mut second = [page];
        allocator.release_sequence(16, &mut second);
        assert_eq!(allocator.snapshot().allocated_pages, 1);
        Ok(())
    }

    #[test]
    fn cached_pages_reduce_future_private_reservation_capacity() -> Result<()> {
        let mut allocator = KvPageAllocator::new(4, KvPageSize::P16)?;
        let reserved = allocator.try_reserve_tokens(16)?;
        let mut table = [u32::MAX; 1];
        allocator.grow_sequence(0, 16, &mut table)?;
        allocator.pin_cached_pages(&[table[0]])?;
        allocator.release_sequence(16, &mut table);
        allocator.release_reservation(reserved);

        assert_eq!(allocator.try_reserve_pages(3)?, 3);
        assert_eq!(
            allocator.try_reserve_pages(1),
            Err(KvAllocationError::LogicalReservationExhausted)
        );
        Ok(())
    }
}
