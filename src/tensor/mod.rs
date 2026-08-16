pub mod shape;

use std::{
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

use anyhow::{Result, ensure};
use cudarc::driver::CudaSlice;

pub use shape::Shape;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct BufferPoolStats {
    pub hits: u64,
    pub misses: u64,
    pub available_elements: usize,
    pub dropped_elements: u64,
    pub internal_fragment_elements: u64,
}

struct BufferPoolState<T> {
    available: Box<[Vec<CudaSlice<T>>]>,
    available_elements: usize,
    hits: u64,
    misses: u64,
    dropped_elements: u64,
    internal_fragment_elements: u64,
}

pub(crate) struct BufferPool<T> {
    state: Mutex<BufferPoolState<T>>,
    max_available_elements: usize,
}

impl<T> BufferPool<T> {
    const PER_SIZE_CLASS_CAPACITY: usize = 16;

    pub(crate) fn new(max_available_elements: usize) -> Self {
        let available = (0..usize::BITS)
            .map(|_| Vec::with_capacity(Self::PER_SIZE_CLASS_CAPACITY))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            state: Mutex::new(BufferPoolState {
                available,
                available_elements: 0,
                hits: 0,
                misses: 0,
                dropped_elements: 0,
                internal_fragment_elements: 0,
            }),
            max_available_elements,
        }
    }

    pub(crate) fn allocation_elements(elements: usize) -> Option<usize> {
        elements.max(1).checked_next_power_of_two()
    }

    pub(crate) fn take(&self, elements: usize) -> Option<CudaSlice<T>> {
        let allocation_elements = Self::allocation_elements(elements)?;
        let first_class = allocation_elements.trailing_zeros() as usize;
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        let storage = state.available[first_class..].iter_mut().find_map(Vec::pop);
        if let Some(storage) = storage {
            let storage_elements = storage.len();
            state.available_elements -= storage_elements;
            state.hits += 1;
            state.internal_fragment_elements = state
                .internal_fragment_elements
                .saturating_add(storage_elements.saturating_sub(elements) as u64);
            Some(storage)
        } else {
            state.misses += 1;
            None
        }
    }

    fn recycle(&self, storage: CudaSlice<T>) {
        let elements = storage.len();
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        let Some(next_available) = state.available_elements.checked_add(elements) else {
            return;
        };
        if next_available > self.max_available_elements {
            state.dropped_elements = state.dropped_elements.saturating_add(elements as u64);
            return;
        }
        let class = elements.trailing_zeros() as usize;
        if class >= state.available.len()
            || state.available[class].len() == Self::PER_SIZE_CLASS_CAPACITY
        {
            state.dropped_elements = state.dropped_elements.saturating_add(elements as u64);
            return;
        }
        state.available[class].push(storage);
        state.available_elements = next_available;
    }

    pub(crate) fn stats(&self) -> BufferPoolStats {
        let state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        BufferPoolStats {
            hits: state.hits,
            misses: state.misses,
            available_elements: state.available_elements,
            dropped_elements: state.dropped_elements,
            internal_fragment_elements: state.internal_fragment_elements,
        }
    }
}

pub struct Tensor<T> {
    storage: ManuallyDrop<CudaSlice<T>>,
    shape: Shape,
    pool: Option<Arc<BufferPool<T>>>,
}

impl<T> Tensor<T> {
    pub fn new(storage: CudaSlice<T>, shape: Shape) -> Result<Self> {
        ensure!(
            storage.len() == shape.numel(),
            "tensor shape/storage mismatch: shape requires {} elements, storage has {}",
            shape.numel(),
            storage.len(),
        );

        Ok(Self {
            storage: ManuallyDrop::new(storage),
            shape,
            pool: None,
        })
    }

    pub(crate) fn new_pooled(
        storage: CudaSlice<T>,
        shape: Shape,
        pool: Arc<BufferPool<T>>,
    ) -> Result<Self> {
        ensure!(
            storage.len() >= shape.numel(),
            "pooled tensor capacity mismatch: shape requires {} elements, storage has {}",
            shape.numel(),
            storage.len(),
        );
        Ok(Self {
            storage: ManuallyDrop::new(storage),
            shape,
            pool: Some(pool),
        })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    pub(crate) fn storage_capacity(&self) -> usize {
        self.storage.len()
    }

    pub fn storage(&self) -> &CudaSlice<T> {
        &self.storage
    }

    pub fn storage_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.storage
    }

    pub fn reshape(mut self, shape: Shape) -> Result<Self> {
        ensure!(
            self.numel() == shape.numel(),
            "reshape changes element count: old_shape={:?}, new_shape={:?}, old_numel={}, new_numel={}",
            self.dims(),
            shape.dims(),
            self.numel(),
            shape.numel()
        );

        self.shape = shape;
        Ok(self)
    }

    pub(crate) fn set_logical_shape(&mut self, shape: Shape) -> Result<()> {
        ensure!(
            shape.numel() <= self.storage.len(),
            "logical shape requires {} elements but storage capacity is {}",
            shape.numel(),
            self.storage.len()
        );
        self.shape = shape;
        Ok(())
    }
}

impl<T> Drop for Tensor<T> {
    fn drop(&mut self) {
        // SAFETY: storage is taken exactly once here. Tensor never exposes an
        // operation that moves it out, and ManuallyDrop suppresses a second drop.
        let storage = unsafe { ManuallyDrop::take(&mut self.storage) };
        if let Some(pool) = &self.pool {
            pool.recycle(storage);
        } else {
            drop(storage);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BufferPool;

    #[test]
    fn pool_uses_bounded_power_of_two_size_classes() {
        assert_eq!(BufferPool::<u8>::allocation_elements(1), Some(1));
        assert_eq!(BufferPool::<u8>::allocation_elements(3), Some(4));
        assert_eq!(BufferPool::<u8>::allocation_elements(65_537), Some(131_072));
        assert_eq!(BufferPool::<u8>::allocation_elements(usize::MAX), None);
    }
}
