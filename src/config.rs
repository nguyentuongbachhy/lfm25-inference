use std::{fs, path::Path};

use anyhow::{Context as _, Result, ensure};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Lfm2Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    #[serde(rename = "conv_L_cache")]
    pub conv_l_cache: usize,
    pub layer_types: Vec<String>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub block_auto_adjust_ff_dim: bool,
    pub block_ffn_dim_multiplier: f32,
    pub block_multiple_of: usize,
}

impl Lfm2Config {
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let data = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        let config: Self = serde_json::from_slice(&data)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn effective_intermediate_size(&self) -> usize {
        if !self.block_auto_adjust_ff_dim {
            return self.intermediate_size;
        }
        let adjusted = (2 * self.intermediate_size / 3) as f32 * self.block_ffn_dim_multiplier;
        let adjusted = adjusted as usize;
        adjusted.div_ceil(self.block_multiple_of) * self.block_multiple_of
    }

    pub fn is_attention_layer(&self, layer: usize) -> bool {
        self.layer_types[layer] == "full_attention"
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.hidden_size == 2048,
            "runtime currently supports LFM2 hidden_size=2048"
        );
        ensure!(
            self.num_attention_heads == 32,
            "runtime currently supports 32 Q heads"
        );
        ensure!(
            self.num_key_value_heads == 8,
            "runtime currently supports 8 KV heads"
        );
        ensure!(
            self.vocab_size == 65536,
            "runtime currently supports vocab_size=65536"
        );
        ensure!(self.bos_token_id == 1, "unexpected LFM2 BOS token id");
        ensure!(self.pad_token_id == 0, "unexpected LFM2 padding token id");
        ensure!(
            self.head_dim() == 64,
            "runtime currently supports head_dim=64"
        );
        ensure!(
            self.conv_l_cache == 3,
            "runtime currently supports convolution width 3"
        );
        ensure!(
            self.layer_types.len() == self.num_hidden_layers,
            "layer_types length does not match num_hidden_layers"
        );
        ensure!(
            self.effective_intermediate_size() == 8192,
            "runtime expected effective FFN width 8192, got {}",
            self.effective_intermediate_size()
        );
        ensure!(
            self.norm_eps.is_finite() && self.norm_eps > 0.0,
            "invalid norm epsilon"
        );
        Ok(())
    }
}
