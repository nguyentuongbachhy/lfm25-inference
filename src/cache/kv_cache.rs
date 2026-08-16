use anyhow::{Context as _, Result, ensure};
use half::bf16;

use crate::{
    cuda::CudaRuntime,
    tensor::{Shape, Tensor},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPageSize {
    P16,
    P32,
}

impl KvPageSize {
    pub fn value(self) -> usize {
        match self {
            Self::P16 => 16,
            Self::P32 => 32,
        }
    }
}

pub struct PagedKvCache {
    page_size: KvPageSize,
    num_pages: usize,
    block_table: Tensor<u32>,
    key: Tensor<bf16>,
    value: Tensor<bf16>,
}

impl PagedKvCache {
    pub fn new(runtime: &CudaRuntime, capacity: usize, page_size: KvPageSize) -> Result<Self> {
        ensure!(capacity > 0, "KV cache capacity must be positive");
        let size = page_size.value();
        let num_pages = capacity.div_ceil(size);
        let block_table_host: Vec<u32> = (0..num_pages)
            .map(u32::try_from)
            .collect::<std::result::Result<_, _>>()
            .context("KV cache page count exceeds u32")?;
        Self::with_block_table(runtime, page_size, num_pages, &block_table_host)
    }

    pub(crate) fn with_block_table(
        runtime: &CudaRuntime,
        page_size: KvPageSize,
        num_physical_pages: usize,
        block_table_host: &[u32],
    ) -> Result<Self> {
        ensure!(
            num_physical_pages > 0,
            "KV cache requires at least one physical page"
        );
        ensure!(
            !block_table_host.is_empty(),
            "KV cache block table must not be empty"
        );
        for &physical_page in block_table_host {
            let physical_page_index =
                usize::try_from(physical_page).context("physical page index does not fit usize")?;
            ensure!(
                physical_page_index < num_physical_pages,
                "block table physical page {physical_page} exceeds pool size {num_physical_pages}"
            );
        }
        let size = page_size.value();
        let shape = Shape::new([num_physical_pages, 8, size, 64]);
        let block_table = runtime.upload(block_table_host, Shape::new([block_table_host.len()]))?;

        Ok(Self {
            page_size,
            num_pages: num_physical_pages,
            block_table,
            key: runtime.zeros::<bf16>(shape.clone())?,
            value: runtime.zeros::<bf16>(shape)?,
        })
    }

    pub fn page_size(&self) -> KvPageSize {
        self.page_size
    }
    pub fn num_pages(&self) -> usize {
        self.num_pages
    }
    pub fn block_table(&self) -> &Tensor<u32> {
        &self.block_table
    }
    pub fn key(&self) -> &Tensor<bf16> {
        &self.key
    }
    pub fn value(&self) -> &Tensor<bf16> {
        &self.value
    }

    pub fn write_lfm2(
        &mut self,
        runtime: &CudaRuntime,
        key: &Tensor<bf16>,
        value: &Tensor<bf16>,
        slot_mapping: &Tensor<i64>,
    ) -> Result<()> {
        ensure!(
            key.rank() == 3 && key.dims()[1..] == [8, 64],
            "LFM2 K must have shape [N,8,64], got {:?}",
            key.dims()
        );
        ensure!(
            value.shape() == key.shape(),
            "LFM2 K/V shape mismatch: K={:?}, V={:?}",
            key.dims(),
            value.dims()
        );
        let num_tokens = key.dims()[0];
        ensure!(
            slot_mapping.numel() == num_tokens,
            "slot mapping must contain {num_tokens} entries, got {:?}",
            slot_mapping.dims()
        );

        unsafe {
            runtime
                .kernels()
                .kv_cache()
                .launch_write_lfm2_bf16(
                    runtime.stream(),
                    self.page_size.value(),
                    key.storage(),
                    value.storage(),
                    self.key.storage_mut(),
                    self.value.storage_mut(),
                    slot_mapping.storage(),
                    num_tokens,
                    self.num_pages,
                )
                .context("failed to write paged LFM2 KV cache")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::cuda::testing::readback;

    use super::*;

    fn check_write(page_size: KvPageSize) -> Result<()> {
        let runtime = CudaRuntime::new(0)?;
        let key_host: Vec<bf16> = (0..2 * 8 * 64)
            .map(|index| bf16::from_f32(index as f32 / 64.0))
            .collect();
        let value_host: Vec<bf16> = (0..2 * 8 * 64)
            .map(|index| bf16::from_f32(-(index as f32) / 64.0))
            .collect();
        let key = runtime.upload(&key_host, Shape::new([2, 8, 64]))?;
        let value = runtime.upload(&value_host, Shape::new([2, 8, 64]))?;
        let size = page_size.value();
        let slots_host = [0i64, (size + 1) as i64];
        let slots = runtime.upload(&slots_host, Shape::new([2]))?;
        let mut cache = PagedKvCache::new(&runtime, size * 2, page_size)?;

        cache.write_lfm2(&runtime, &key, &value, &slots)?;
        let actual_key = readback(&runtime, cache.key())?;
        let actual_value = readback(&runtime, cache.value())?;

        for token in 0..2 {
            let slot = [0usize, size + 1][token];
            let page = slot / size;
            let offset = slot % size;
            for head in 0..8 {
                for dim in 0..64 {
                    let source = (token * 8 + head) * 64 + dim;
                    let destination = ((page * 8 + head) * size + offset) * 64 + dim;
                    assert_eq!(
                        actual_key[destination].to_bits(),
                        key_host[source].to_bits()
                    );
                    assert_eq!(
                        actual_value[destination].to_bits(),
                        value_host[source].to_bits()
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn paged_kv_write_ps16_preserves_bits_and_layout() -> Result<()> {
        check_write(KvPageSize::P16)
    }

    #[test]
    fn paged_kv_write_ps32_preserves_bits_and_layout() -> Result<()> {
        check_write(KvPageSize::P32)
    }
}
