use std::{collections::HashMap, fs::File, path::Path};

use anyhow::{Context as _, Result, ensure};
use half::bf16;
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};

use crate::{
    cuda::CudaRuntime,
    tensor::{Shape, Tensor},
};

pub struct WeightStore {
    tensors: HashMap<String, Tensor<bf16>>,
}

impl WeightStore {
    pub fn load(runtime: &CudaRuntime, model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("model.safetensors");
        let file =
            File::open(&path).with_context(|| format!("failed to open {}", path.display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("failed to map {}", path.display()))?;
        let archive = SafeTensors::deserialize(&mmap)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let mut entries = archive.tensors();
        entries.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        let total = entries.len();
        let mut tensors = HashMap::with_capacity(total);

        for (index, (name, view)) in entries.into_iter().enumerate() {
            ensure!(
                view.dtype() == Dtype::BF16,
                "unsupported dtype {:?} for tensor {name}",
                view.dtype()
            );
            let bytes = view.data();
            ensure!(bytes.len() % 2 == 0, "invalid BF16 byte length for {name}");
            let host: Vec<bf16> = bytes
                .chunks_exact(2)
                .map(|pair| bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])))
                .collect();
            let tensor = runtime
                .upload(&host, Shape::new(view.shape().iter().copied()))
                .with_context(|| format!("failed to upload weight {name}"))?;
            tensors.insert(name, tensor);

            if (index + 1) % 16 == 0 || index + 1 == total {
                eprintln!("loaded {}/{} weight tensors", index + 1, total);
            }
        }

        Ok(Self { tensors })
    }

    pub fn take(&mut self, name: &str) -> Result<Tensor<bf16>> {
        self.tensors
            .remove(name)
            .with_context(|| format!("missing model weight {name}"))
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}
