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

    pub fn encode_chat_prompt(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        let mut formatted = String::from("<|startoftext|>");
        for message in messages {
            formatted.push_str("<|im_start|>");
            formatted.push_str(&message.role);
            formatted.push('\n');
            formatted.push_str(&message.text());
            formatted.push_str("<|im_end|>\n");
        }
        formatted.push_str("<|im_start|>assistant\n");
        let encoding = self
            .inner
            .encode(formatted, false)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .context("failed to tokenize chat prompt")?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .context("failed to decode generated tokens")
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    pub fn text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Parts(parts) => {
                let mut full = String::new();
                for part in parts {
                    if let ContentPart::Text { text } = part {
                        full.push_str(text);
                    }
                }
                full
            }
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContent,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: MessageContent::Text(content.into()),
            name: None,
        }
    }

    pub fn text(&self) -> String {
        self.content.text()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_content_deserialization() {
        let msg_constructed = ChatMessage::new("user", "constructed");
        assert_eq!(msg_constructed.role, "user");
        assert_eq!(msg_constructed.text(), "constructed");

        let json_str = r#"{"role":"user","content":"hello"}"#;
        let msg: ChatMessage = serde_json::from_str(json_str).unwrap();
        assert_eq!(msg.role, "user");
        assert_eq!(msg.text(), "hello");

        let json_parts = r#"{"role":"user","content":[{"type":"text","text":"hello "},{"type":"text","text":"world"}]}"#;
        let msg_parts: ChatMessage = serde_json::from_str(json_parts).unwrap();
        assert_eq!(msg_parts.text(), "hello world");
    }
}
