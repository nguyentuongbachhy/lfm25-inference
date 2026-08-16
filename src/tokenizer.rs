use std::path::Path;

use anyhow::{Context as _, Result};
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct Lfm2Tokenizer {
    inner: Tokenizer,
}

impl Lfm2Tokenizer {
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("tokenizer.json");
        let inner = Tokenizer::from_file(&path)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .with_context(|| format!("failed to load {}", path.display()))?;
        Ok(Self { inner })
    }

    pub fn encode_user_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        let formatted =
            format!("<|startoftext|><|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
        let encoding = self
            .inner
            .encode(formatted, false)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .context("failed to tokenize prompt")?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .context("failed to tokenize corpus text")?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .context("failed to decode generated tokens")
    }
}
