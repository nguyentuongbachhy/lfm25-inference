use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::{
    engine::{GenerationMetrics, GenerationOptions, ServingCompletion},
    generation::{DEFAULT_SAMPLING_SEED, SamplingConfig},
    tokenizer::ChatMessage,
};

pub(crate) const DEFAULT_MODEL_NAME: &str = "LFM2.5-1.2B-Instruct";

#[derive(Debug)]
pub(crate) struct RouteResponse {
    pub status: &'static str,
    pub content_type: &'static str,
    pub body: String,
}

#[derive(Serialize)]
struct ErrorEnvelope<'a> {
    error: ErrorBody<'a>,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    message: &'a str,
    #[serde(rename = "type")]
    error_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    param: Option<&'static str>,
    code: u16,
}

#[derive(Serialize)]
pub(crate) struct ModelListResponse {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

#[derive(Serialize, Clone)]
pub(crate) struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
    pub root: String,
    pub parent: Option<String>,
    pub max_model_len: usize,
    pub permission: Vec<ModelPermission>,
}

#[derive(Serialize, Clone)]
pub(crate) struct ModelPermission {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub allow_create_engine: bool,
    pub allow_sampling: bool,
    pub allow_logprobs: bool,
    pub allow_search_indices: bool,
    pub allow_view: bool,
    pub allow_fine_tuning: bool,
    pub organization: &'static str,
    pub group: Option<String>,
    pub is_blocking: bool,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub(crate) enum StopCondition {
    Single(String),
    Multiple(Vec<String>),
}

impl StopCondition {
    pub fn as_vec(&self) -> Vec<&str> {
        match self {
            Self::Single(s) => vec![s.as_str()],
            Self::Multiple(list) => list.iter().map(|s| s.as_str()).collect(),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub max_new_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<StopCondition>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    metrics: Option<GenerationMetrics>,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatResponseMessage,
    logprobs: Option<()>,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct ChatResponseMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub(crate) enum CompletionPrompt {
    Single(String),
    Multiple(Vec<String>),
}

impl CompletionPrompt {
    pub fn into_first(self) -> Result<String, &'static str> {
        match self {
            Self::Single(text) => Ok(text),
            Self::Multiple(mut list) => {
                if list.is_empty() {
                    Err("prompt array must not be empty")
                } else {
                    Ok(list.remove(0))
                }
            }
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: CompletionPrompt,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub max_new_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<StopCondition>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    metrics: Option<GenerationMetrics>,
}

#[derive(Serialize)]
struct Choice {
    text: String,
    index: usize,
    logprobs: Option<()>,
    finish_reason: &'static str,
}

#[derive(Serialize, Clone)]
pub(crate) struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize)]
struct ChatChunkResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
}

#[derive(Serialize)]
struct ChatChunkChoice {
    index: usize,
    delta: ChatChunkDelta,
    logprobs: Option<()>,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
struct ChatChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Serialize)]
struct TextChunkResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<TextChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
}

#[derive(Serialize)]
struct TextChunkChoice {
    index: usize,
    text: String,
    logprobs: Option<()>,
    finish_reason: Option<&'static str>,
}

pub(crate) fn health() -> RouteResponse {
    json_response("200 OK", &serde_json::json!({ "status": "ok" }))
}

pub(crate) fn version() -> RouteResponse {
    json_response(
        "200 OK",
        &serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"),
        }),
    )
}

pub(crate) fn cors_preflight() -> RouteResponse {
    RouteResponse {
        status: "204 No Content",
        content_type: "text/plain",
        body: String::new(),
    }
}

pub(crate) fn models_list(model_name: &str) -> RouteResponse {
    let model = build_model_object(model_name);
    json_response(
        "200 OK",
        &ModelListResponse {
            object: "list",
            data: vec![model],
        },
    )
}

pub(crate) fn model_retrieve(model_name: &str, requested_id: &str) -> RouteResponse {
    if requested_id == model_name || requested_id == "default" {
        json_response("200 OK", &build_model_object(model_name))
    } else {
        error_response(
            "404 Not Found",
            &format!("The model '{requested_id}' does not exist"),
            "invalid_request_error",
            404,
            Some("model"),
        )
    }
}

fn build_model_object(model_name: &str) -> ModelObject {
    let created = current_timestamp();
    ModelObject {
        id: model_name.to_string(),
        object: "model",
        created,
        owned_by: "liquid",
        root: model_name.to_string(),
        parent: None,
        max_model_len: 4096,
        permission: vec![ModelPermission {
            id: format!("modelperm-{}", model_name.to_lowercase()),
            object: "model_permission",
            created,
            allow_create_engine: false,
            allow_sampling: true,
            allow_logprobs: true,
            allow_search_indices: false,
            allow_view: true,
            allow_fine_tuning: false,
            organization: "*",
            group: None,
            is_blocking: false,
        }],
    }
}

pub(crate) struct ParsedChatRequest {
    pub messages: Vec<ChatMessage>,
    pub options: GenerationOptions,
    pub model: String,
    pub stop: Option<StopCondition>,
    pub stream: bool,
}

pub(crate) fn parse_chat_completion(
    body: &[u8],
    default_model: &str,
) -> Result<ParsedChatRequest, RouteResponse> {
    let request: ChatCompletionRequest = match serde_json::from_slice(body) {
        Ok(req) => req,
        Err(err) => {
            return Err(error_response(
                "400 Bad Request",
                &format!("invalid JSON payload: {err}"),
                "invalid_request_error",
                400,
                None,
            ));
        }
    };
    if request.messages.is_empty() {
        return Err(error_response(
            "400 Bad Request",
            "messages must not be empty",
            "invalid_request_error",
            400,
            Some("messages"),
        ));
    }
    let max_tokens = match extract_max_tokens(
        request.max_tokens,
        request.max_completion_tokens,
        request.max_new_tokens,
    ) {
        Ok(val) => val,
        Err(msg) => {
            return Err(error_response(
                "400 Bad Request",
                msg,
                "invalid_request_error",
                400,
                Some("max_tokens"),
            ));
        }
    };
    let sampling = match build_sampling_config(
        request.temperature,
        request.top_p,
        request.top_k,
        request.repetition_penalty,
        request.seed,
    ) {
        Ok(cfg) => cfg,
        Err(msg) => {
            return Err(error_response(
                "400 Bad Request",
                &msg,
                "invalid_request_error",
                400,
                None,
            ));
        }
    };
    let model = request.model.unwrap_or_else(|| default_model.to_string());
    Ok(ParsedChatRequest {
        messages: request.messages,
        options: GenerationOptions {
            max_new_tokens: max_tokens,
            sampling,
            speculative_draft: 0,
        },
        model,
        stop: request.stop,
        stream: request.stream.unwrap_or(false),
    })
}

pub(crate) struct ParsedCompletionRequest {
    pub prompt: String,
    pub options: GenerationOptions,
    pub model: String,
    pub stop: Option<StopCondition>,
    pub stream: bool,
}

pub(crate) fn parse_completion(
    body: &[u8],
    default_model: &str,
) -> Result<ParsedCompletionRequest, RouteResponse> {
    let request: CompletionRequest = match serde_json::from_slice(body) {
        Ok(req) => req,
        Err(err) => {
            return Err(error_response(
                "400 Bad Request",
                &format!("invalid JSON payload: {err}"),
                "invalid_request_error",
                400,
                None,
            ));
        }
    };
    let prompt = match request.prompt.into_first() {
        Ok(p) => p,
        Err(err_msg) => {
            return Err(error_response(
                "400 Bad Request",
                err_msg,
                "invalid_request_error",
                400,
                Some("prompt"),
            ));
        }
    };
    let max_tokens = match extract_max_tokens(
        request.max_tokens,
        request.max_completion_tokens,
        request.max_new_tokens,
    ) {
        Ok(val) => val,
        Err(msg) => {
            return Err(error_response(
                "400 Bad Request",
                msg,
                "invalid_request_error",
                400,
                Some("max_tokens"),
            ));
        }
    };
    let sampling = match build_sampling_config(
        request.temperature,
        request.top_p,
        request.top_k,
        request.repetition_penalty,
        request.seed,
    ) {
        Ok(cfg) => cfg,
        Err(msg) => {
            return Err(error_response(
                "400 Bad Request",
                &msg,
                "invalid_request_error",
                400,
                None,
            ));
        }
    };
    let model = request.model.unwrap_or_else(|| default_model.to_string());
    Ok(ParsedCompletionRequest {
        prompt,
        options: GenerationOptions {
            max_new_tokens: max_tokens,
            sampling,
            speculative_draft: 0,
        },
        model,
        stop: request.stop,
        stream: request.stream.unwrap_or(false),
    })
}

pub(crate) fn chat_completion_response(
    request_id: u64,
    model: String,
    completion: ServingCompletion,
    text: String,
    stop: Option<&StopCondition>,
) -> RouteResponse {
    let (cleaned_text, finish_reason) =
        apply_stop_conditions(&text, stop, completion.finish_reason);
    let prompt_tokens = completion.prompt_tokens;
    let completion_tokens = completion.token_ids.len();
    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{:016x}{:08x}", current_timestamp(), request_id),
        object: "chat.completion",
        created: current_timestamp(),
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatResponseMessage {
                role: "assistant",
                content: cleaned_text,
            },
            logprobs: None,
            finish_reason,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
        metrics: Some(completion.metrics),
    };
    json_response("200 OK", &response)
}

pub(crate) fn completion_response(
    request_id: u64,
    model: String,
    completion: ServingCompletion,
    text: String,
    stop: Option<&StopCondition>,
) -> RouteResponse {
    let (cleaned_text, finish_reason) =
        apply_stop_conditions(&text, stop, completion.finish_reason);
    let prompt_tokens = completion.prompt_tokens;
    let completion_tokens = completion.token_ids.len();
    let response = CompletionResponse {
        id: format!("cmpl-{:016x}{:08x}", current_timestamp(), request_id),
        object: "text_completion",
        created: current_timestamp(),
        model,
        choices: vec![Choice {
            text: cleaned_text,
            index: 0,
            logprobs: None,
            finish_reason,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
        metrics: Some(completion.metrics),
    };
    json_response("200 OK", &response)
}

pub(crate) fn chat_completion_sse_stream(
    request_id: u64,
    model: &str,
    completion: &ServingCompletion,
    text: &str,
    stop: Option<&StopCondition>,
) -> RouteResponse {
    let (cleaned_text, finish_reason) = apply_stop_conditions(text, stop, completion.finish_reason);
    let id = format!("chatcmpl-{:016x}{:08x}", current_timestamp(), request_id);
    let created = current_timestamp();
    let prompt_tokens = completion.prompt_tokens;
    let completion_tokens = completion.token_ids.len();
    let usage = Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
    };

    let mut body = String::new();

    // Chunk 1: Role delta
    let chunk1 = ChatChunkResponse {
        id: id.clone(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatChunkDelta {
                role: Some("assistant"),
                content: Some(String::new()),
            },
            logprobs: None,
            finish_reason: None,
        }],
        usage: None,
    };
    append_sse_event(&mut body, &chunk1);

    // Chunk 2: Content delta
    let chunk2 = ChatChunkResponse {
        id: id.clone(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatChunkDelta {
                role: None,
                content: Some(cleaned_text),
            },
            logprobs: None,
            finish_reason: None,
        }],
        usage: None,
    };
    append_sse_event(&mut body, &chunk2);

    // Chunk 3: Final chunk with finish reason and usage
    let chunk3 = ChatChunkResponse {
        id,
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatChunkDelta {
                role: None,
                content: None,
            },
            logprobs: None,
            finish_reason: Some(finish_reason),
        }],
        usage: Some(usage),
    };
    append_sse_event(&mut body, &chunk3);

    // SSE termination
    body.push_str("data: [DONE]\n\n");

    RouteResponse {
        status: "200 OK",
        content_type: "text/event-stream; charset=utf-8",
        body,
    }
}

pub(crate) fn completion_sse_stream(
    request_id: u64,
    model: &str,
    completion: &ServingCompletion,
    text: &str,
    stop: Option<&StopCondition>,
) -> RouteResponse {
    let (cleaned_text, finish_reason) = apply_stop_conditions(text, stop, completion.finish_reason);
    let id = format!("cmpl-{:016x}{:08x}", current_timestamp(), request_id);
    let created = current_timestamp();
    let prompt_tokens = completion.prompt_tokens;
    let completion_tokens = completion.token_ids.len();
    let usage = Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
    };

    let mut body = String::new();

    // Chunk 1: Content
    let chunk1 = TextChunkResponse {
        id: id.clone(),
        object: "text_completion",
        created,
        model: model.to_string(),
        choices: vec![TextChunkChoice {
            index: 0,
            text: cleaned_text,
            logprobs: None,
            finish_reason: None,
        }],
        usage: None,
    };
    append_sse_event(&mut body, &chunk1);

    // Chunk 2: Final with finish reason and usage
    let chunk2 = TextChunkResponse {
        id,
        object: "text_completion",
        created,
        model: model.to_string(),
        choices: vec![TextChunkChoice {
            index: 0,
            text: String::new(),
            logprobs: None,
            finish_reason: Some(finish_reason),
        }],
        usage: Some(usage),
    };
    append_sse_event(&mut body, &chunk2);

    // SSE termination
    body.push_str("data: [DONE]\n\n");

    RouteResponse {
        status: "200 OK",
        content_type: "text/event-stream; charset=utf-8",
        body,
    }
}

fn append_sse_event(buffer: &mut String, value: &impl Serialize) {
    if let Ok(json) = serde_json::to_string(value) {
        buffer.push_str("data: ");
        buffer.push_str(&json);
        buffer.push_str("\n\n");
    }
}

fn extract_max_tokens(
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    max_new_tokens: Option<usize>,
) -> Result<usize, &'static str> {
    let tokens = max_completion_tokens
        .or(max_tokens)
        .or(max_new_tokens)
        .unwrap_or(64);
    if tokens == 0 {
        return Err("max_tokens must be positive");
    }
    Ok(tokens)
}

fn build_sampling_config(
    temperature: Option<f32>,
    _top_p: Option<f32>,
    top_k: Option<usize>,
    repetition_penalty: Option<f32>,
    seed: Option<u64>,
) -> Result<SamplingConfig, String> {
    let config = SamplingConfig {
        temperature: temperature.unwrap_or(0.0),
        top_k: top_k.unwrap_or(50),
        repetition_penalty: repetition_penalty.unwrap_or(1.0),
        seed: seed.unwrap_or(DEFAULT_SAMPLING_SEED),
    };
    config.validate().map_err(|err| err.to_string())
}

fn apply_stop_conditions(
    text: &str,
    stop: Option<&StopCondition>,
    original_finish_reason: &'static str,
) -> (String, &'static str) {
    let mut cleaned = text;

    // Strip ChatML special tokens if present at end of detokenized string
    for special in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(stripped) = cleaned.strip_suffix(special) {
            cleaned = stripped;
        }
    }

    if let Some(stop_cond) = stop {
        let mut earliest_pos = None;
        for stop_word in stop_cond.as_vec() {
            if stop_word.is_empty() {
                continue;
            }
            if let Some(pos) = cleaned.find(stop_word) {
                match earliest_pos {
                    None => earliest_pos = Some(pos),
                    Some(cur) if pos < cur => earliest_pos = Some(pos),
                    _ => {}
                }
            }
        }
        if let Some(pos) = earliest_pos {
            return (cleaned[..pos].to_string(), "stop");
        }
    }

    (cleaned.to_string(), original_finish_reason)
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn json_response(status: &'static str, value: &impl Serialize) -> RouteResponse {
    match serde_json::to_string(value) {
        Ok(body) => RouteResponse {
            status,
            content_type: "application/json",
            body,
        },
        Err(error) => error_response(
            "500 Internal Server Error",
            &format!("JSON encode failed: {error}"),
            "internal_error",
            500,
            None,
        ),
    }
}

pub(crate) fn error_response(
    status: &'static str,
    message: &str,
    error_type: &'static str,
    code: u16,
    param: Option<&'static str>,
) -> RouteResponse {
    let envelope = ErrorEnvelope {
        error: ErrorBody {
            message,
            error_type,
            param,
            code,
        },
    };
    let body = serde_json::to_string(&envelope)
        .unwrap_or_else(|_| "{\"error\":{\"message\":\"internal error\",\"type\":\"internal_error\",\"code\":500}}".to_string());
    RouteResponse {
        status,
        content_type: "application/json",
        body,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_request_parsing() {
        let json = r#"{
            "model": "LFM2.5-1.2B-Instruct",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 128,
            "temperature": 0.5,
            "stop": ["\n"]
        }"#;
        let parsed = parse_chat_completion(json.as_bytes(), "default-model").unwrap();
        assert_eq!(parsed.model, "LFM2.5-1.2B-Instruct");
        assert_eq!(parsed.messages.len(), 2);
        assert_eq!(parsed.options.max_new_tokens, 128);
        assert_eq!(parsed.options.sampling.temperature, 0.5);
        assert_eq!(
            parsed.stop,
            Some(StopCondition::Multiple(vec!["\n".to_string()]))
        );
        assert!(!parsed.stream);

        // Test single string stop condition
        let json_single_stop = r#"{
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": "\n"
        }"#;
        let parsed_single = parse_chat_completion(json_single_stop.as_bytes(), "default-model").unwrap();
        assert_eq!(
            parsed_single.stop,
            Some(StopCondition::Single("\n".to_string()))
        );
    }

    #[test]
    fn test_completion_request_parsing() {
        let json = r#"{
            "prompt": "Hello",
            "max_tokens": 64,
            "stream": true
        }"#;
        let parsed = parse_completion(json.as_bytes(), "default-model").unwrap();
        assert_eq!(parsed.prompt, "Hello");
        assert_eq!(parsed.options.max_new_tokens, 64);
        assert!(parsed.stream);
    }

    #[test]
    fn test_stop_condition_application() {
        let text = "Hello world! Stop here. More text";
        let stop = StopCondition::Single("Stop here".to_string());
        let (res, finish) = apply_stop_conditions(text, Some(&stop), "length");
        assert_eq!(res, "Hello world! ");
        assert_eq!(finish, "stop");
    }

    #[test]
    fn test_error_response_formatting() {
        let resp = error_response(
            "400 Bad Request",
            "test error",
            "invalid_request_error",
            400,
            Some("prompt"),
        );
        assert_eq!(resp.status, "400 Bad Request");
        assert_eq!(resp.content_type, "application/json");
        assert!(resp.body.contains("\"message\":\"test error\""));
        assert!(resp.body.contains("\"type\":\"invalid_request_error\""));
        assert!(resp.body.contains("\"code\":400"));
        assert!(resp.body.contains("\"param\":\"prompt\""));
    }

    #[test]
    fn test_models_list_response() {
        let resp = models_list("LFM2.5-1.2B-Instruct");
        assert_eq!(resp.status, "200 OK");
        assert!(resp.body.contains("\"id\":\"LFM2.5-1.2B-Instruct\""));
        assert!(resp.body.contains("\"object\":\"list\""));
    }
}

