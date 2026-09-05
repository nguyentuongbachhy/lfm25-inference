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

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub(crate) enum ResponseFormat {
    JsonSchema {
        r#type: String,
        json_schema: JsonSchemaSpec,
    },
    TypeOnly {
        r#type: String,
    },
    RawSchema(serde_json::Value),
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub(crate) struct JsonSchemaSpec {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub strict: Option<bool>,
    pub schema: serde_json::Value,
}

#[derive(Deserialize)]
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
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub format: Option<serde_json::Value>,
    #[serde(default)]
    pub options: Option<serde_json::Value>,
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
        max_model_len: 32_768,
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
    pub is_structured_json: bool,
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
    let options_num_predict = request
        .options
        .as_ref()
        .and_then(|opt| opt.get("num_predict"))
        .and_then(|np| np.as_u64())
        .map(|np| np as usize);

    let max_tokens = match extract_max_tokens(
        request.max_tokens,
        request.max_completion_tokens,
        request.max_new_tokens,
        options_num_predict,
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
    let temperature = request.temperature.or_else(|| {
        request
            .options
            .as_ref()
            .and_then(|opt| opt.get("temperature"))
            .and_then(|t| t.as_f64())
            .map(|t| t as f32)
    });
    let top_p = request.top_p.or_else(|| {
        request
            .options
            .as_ref()
            .and_then(|opt| opt.get("top_p"))
            .and_then(|t| t.as_f64())
            .map(|t| t as f32)
    });
    let top_k = request.top_k.or_else(|| {
        request
            .options
            .as_ref()
            .and_then(|opt| opt.get("top_k"))
            .and_then(|t| t.as_u64())
            .map(|t| t as usize)
    });
    let seed = request.seed.or_else(|| {
        request
            .options
            .as_ref()
            .and_then(|opt| opt.get("seed"))
            .and_then(|t| t.as_u64())
    });
    let sampling =
        match build_sampling_config(temperature, top_p, top_k, request.repetition_penalty, seed) {
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
    let constraint =
        extract_schema_constraint(request.response_format.as_ref(), request.format.as_ref());
    let is_structured_json = constraint.is_some();
    let mut messages = request.messages;
    if let Some(schema_prompt) = constraint {
        if let Some(first) = messages.first_mut() {
            if first.role == "system" {
                let mut content = first.text();
                content.push_str(&schema_prompt);
                first.content = crate::tokenizer::MessageContent::Text(content);
            } else {
                messages.insert(
                    0,
                    ChatMessage::new("system", schema_prompt.trim().to_string()),
                );
            }
        } else {
            messages.push(ChatMessage::new("system", schema_prompt.trim().to_string()));
        }
    }
    Ok(ParsedChatRequest {
        messages,
        options: GenerationOptions {
            max_new_tokens: max_tokens,
            sampling,
            speculative_draft: 0,
        },
        model,
        stop: request.stop,
        stream: request.stream.unwrap_or(false),
        is_structured_json,
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
        None,
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
    is_structured_json: bool,
) -> RouteResponse {
    let (cleaned_text, finish_reason) =
        apply_stop_conditions(&text, stop, completion.finish_reason);
    let final_content = if is_structured_json {
        clean_structured_output(&cleaned_text)
    } else {
        cleaned_text
    };
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
                content: final_content,
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
    is_structured_json: bool,
) -> RouteResponse {
    let (cleaned_text, finish_reason) = apply_stop_conditions(text, stop, completion.finish_reason);
    let final_content = if is_structured_json {
        clean_structured_output(&cleaned_text)
    } else {
        cleaned_text
    };
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
                content: Some(final_content),
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
    buffer.push_str(&format_sse_event(value));
}

pub(crate) fn format_sse_chat_role_chunk(id: &str, model: &str, created: u64) -> String {
    let chunk = ChatChunkResponse {
        id: id.to_string(),
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
    format_sse_event(&chunk)
}

pub(crate) fn format_sse_chat_content_chunk(
    id: &str,
    model: &str,
    created: u64,
    delta: &str,
) -> String {
    let chunk = ChatChunkResponse {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatChunkDelta {
                role: None,
                content: Some(delta.to_string()),
            },
            logprobs: None,
            finish_reason: None,
        }],
        usage: None,
    };
    format_sse_event(&chunk)
}

pub(crate) fn format_sse_chat_finish_chunk(
    id: &str,
    model: &str,
    created: u64,
    finish_reason: &'static str,
    usage: Usage,
) -> String {
    let chunk = ChatChunkResponse {
        id: id.to_string(),
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
    format_sse_event(&chunk)
}

pub(crate) fn format_sse_text_content_chunk(
    id: &str,
    model: &str,
    created: u64,
    delta: &str,
) -> String {
    let chunk = TextChunkResponse {
        id: id.to_string(),
        object: "text_completion",
        created,
        model: model.to_string(),
        choices: vec![TextChunkChoice {
            index: 0,
            text: delta.to_string(),
            logprobs: None,
            finish_reason: None,
        }],
        usage: None,
    };
    format_sse_event(&chunk)
}

pub(crate) fn format_sse_text_finish_chunk(
    id: &str,
    model: &str,
    created: u64,
    finish_reason: &'static str,
    usage: Usage,
) -> String {
    let chunk = TextChunkResponse {
        id: id.to_string(),
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
    format_sse_event(&chunk)
}

pub(crate) fn format_sse_event(value: &impl Serialize) -> String {
    match serde_json::to_string(value) {
        Ok(json) => format!("data: {json}\n\n"),
        Err(_) => String::new(),
    }
}

pub(crate) fn check_stop_match(text: &str, stop_words: &[&str]) -> Option<usize> {
    let mut earliest = None;
    for &word in stop_words {
        if word.is_empty() {
            continue;
        }
        if let Some(pos) = text.find(word) {
            match earliest {
                None => earliest = Some(pos),
                Some(p) if pos < p => earliest = Some(pos),
                _ => {}
            }
        }
    }
    earliest
}

pub(crate) fn longest_stop_prefix_len(text: &str, stop_words: &[&str]) -> usize {
    let mut max_prefix = 0;
    for &word in stop_words {
        if word.is_empty() {
            continue;
        }
        let check_len = text.len().min(word.len());
        for len in (1..=check_len).rev() {
            if text.ends_with(&word[..len]) {
                if len > max_prefix {
                    max_prefix = len;
                }
                break;
            }
        }
    }
    max_prefix
}

fn extract_max_tokens(
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    max_new_tokens: Option<usize>,
    options_num_predict: Option<usize>,
) -> Result<usize, &'static str> {
    let tokens = max_completion_tokens
        .or(max_tokens)
        .or(max_new_tokens)
        .or(options_num_predict)
        .unwrap_or(128);
    if tokens == 0 {
        return Err("max_tokens must be positive");
    }
    Ok(tokens)
}

fn extract_schema_constraint(
    response_format: Option<&ResponseFormat>,
    format: Option<&serde_json::Value>,
) -> Option<String> {
    if let Some(rf) = response_format {
        match rf {
            ResponseFormat::JsonSchema { r#type, json_schema } if r#type == "json_schema" => {
                let schema_str = serde_json::to_string_pretty(&json_schema.schema)
                    .unwrap_or_else(|_| json_schema.schema.to_string());
                Some(format!(
                    "\n\n[RESPONSE FORMAT INSTRUCTION]\nYou must respond strictly with a single valid JSON object adhering to the following JSON schema. Do not enclose the output in markdown code blocks or backticks, and do not include any commentary outside the JSON:\n{schema_str}"
                ))
            }
            ResponseFormat::TypeOnly { r#type } if r#type == "json_object" => {
                Some(
                    "\n\n[RESPONSE FORMAT INSTRUCTION]\nYou must respond strictly with a single valid JSON object. Do not enclose the output in markdown code blocks or backticks, and do not include any commentary outside the JSON."
                        .to_string(),
                )
            }
            ResponseFormat::RawSchema(val) => {
                if let Some(s) = val.as_str() {
                    if s == "json" {
                        return Some(
                            "\n\n[RESPONSE FORMAT INSTRUCTION]\nYou must respond strictly with a single valid JSON object. Do not enclose the output in markdown code blocks or backticks, and do not include any commentary outside the JSON."
                                .to_string(),
                        );
                    }
                } else if val.is_object() {
                    let schema_str = serde_json::to_string_pretty(val)
                        .unwrap_or_else(|_| val.to_string());
                    return Some(format!(
                        "\n\n[RESPONSE FORMAT INSTRUCTION]\nYou must respond strictly with a single valid JSON object adhering to the following JSON schema. Do not enclose the output in markdown code blocks or backticks, and do not include any commentary outside the JSON:\n{schema_str}"
                    ));
                }
                None
            }
            _ => None,
        }
    } else if let Some(fmt) = format {
        if let Some(s) = fmt.as_str() {
            if s == "json" {
                Some(
                    "\n\n[RESPONSE FORMAT INSTRUCTION]\nYou must respond strictly with a single valid JSON object. Do not enclose the output in markdown code blocks or backticks, and do not include any commentary outside the JSON."
                        .to_string(),
                )
            } else {
                None
            }
        } else if fmt.is_object() {
            let schema_str = serde_json::to_string_pretty(fmt).unwrap_or_else(|_| fmt.to_string());
            Some(format!(
                "\n\n[RESPONSE FORMAT INSTRUCTION]\nYou must respond strictly with a single valid JSON object adhering to the following JSON schema. Do not enclose the output in markdown code blocks or backticks, and do not include any commentary outside the JSON:\n{schema_str}"
            ))
        } else {
            None
        }
    } else {
        None
    }
}

fn clean_structured_output(text: &str) -> String {
    let trimmed = text.trim();
    // 1. If wrapped in markdown ```json ... ``` or ``` ... ```
    if let Some(stripped) = trimmed.strip_prefix("```") {
        let after_fence = if let Some(rest) = stripped.strip_prefix("json") {
            rest
        } else {
            stripped
        };
        let inner = if let Some(end_idx) = after_fence.rfind("```") {
            &after_fence[..end_idx]
        } else {
            after_fence
        };
        let cleaned = inner.trim();
        if serde_json::from_str::<serde_json::Value>(cleaned).is_ok() {
            return cleaned.to_string();
        }
    }
    // 2. If the trimmed text is already valid JSON
    if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
        return trimmed.to_string();
    }
    // 3. Try to locate the outermost matching '{' ... '}' or '[' ... ']'
    if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) {
        let candidate = trimmed[start..=end].trim();
        if start < end && serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
            return candidate.to_string();
        }
    }
    if let (Some(start), Some(end)) = (trimmed.find('['), trimmed.rfind(']')) {
        let candidate = trimmed[start..=end].trim();
        if start < end && serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
            return candidate.to_string();
        }
    }
    // Fallback to trimmed text
    trimmed.to_string()
}

fn build_sampling_config(
    temperature: Option<f32>,
    _top_p: Option<f32>,
    top_k: Option<usize>,
    repetition_penalty: Option<f32>,
    seed: Option<u64>,
) -> Result<SamplingConfig, String> {
    // Continuous GPU serving utilizes high-performance fused argmax decode kernels (greedy).
    // Standard OpenAI clients (Cursor, OpenWebUI, Python openai SDK, LangChain) frequently
    // send non-zero default temperatures (such as 0.7 or 1.0). To ensure total client
    // compatibility without unexpected 500/400 errors, we safely clamp to greedy (0.0).
    let _ = (temperature, repetition_penalty);
    let config = SamplingConfig {
        temperature: 0.0,
        top_k: top_k.unwrap_or(50),
        repetition_penalty: 1.0,
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

pub(crate) fn current_timestamp() -> u64 {
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
    let body = serde_json::to_string(&envelope).unwrap_or_else(|_| {
        "{\"error\":{\"message\":\"internal error\",\"type\":\"internal_error\",\"code\":500}}"
            .to_string()
    });
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
        assert_eq!(parsed.options.sampling.temperature, 0.0);
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
        let parsed_single =
            parse_chat_completion(json_single_stop.as_bytes(), "default-model").unwrap();
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

    #[test]
    fn test_openai_json_schema_response_format() {
        let json = r#"{
            "model": "LFM2.5-1.2B-Instruct",
            "messages": [
                {"role": "user", "content": "Return json"}
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "enterprise_answer",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answerable": {"type": "boolean"},
                            "answer": {"type": "string"}
                        },
                        "required": ["answerable", "answer"]
                    }
                }
            }
        }"#;
        let parsed = parse_chat_completion(json.as_bytes(), "default-model").unwrap();
        assert!(parsed.is_structured_json);
        assert_eq!(parsed.messages.len(), 2);
        assert_eq!(parsed.messages[0].role, "system");
        assert!(
            parsed.messages[0]
                .text()
                .contains("[RESPONSE FORMAT INSTRUCTION]")
        );
        assert!(parsed.messages[0].text().contains("answerable"));
    }

    #[test]
    fn test_openai_json_object_response_format() {
        let json = r#"{
            "messages": [
                {"role": "user", "content": "Return json"}
            ],
            "response_format": {"type": "json_object"}
        }"#;
        let parsed = parse_chat_completion(json.as_bytes(), "default-model").unwrap();
        assert!(parsed.is_structured_json);
        assert_eq!(parsed.messages[0].role, "system");
        assert!(
            parsed.messages[0]
                .text()
                .contains("[RESPONSE FORMAT INSTRUCTION]")
        );
    }

    #[test]
    fn test_ollama_format_and_options_payload() {
        let json = r#"{
            "model": "LFM2.5-1.2B-Instruct",
            "stream": false,
            "format": {
                "type": "object",
                "properties": {
                    "answerable": {"type": "boolean"},
                    "answer": {"type": "string"},
                    "selected_chunk_ids": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["answerable", "answer", "selected_chunk_ids"]
            },
            "keep_alive": "5m",
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 128},
            "messages": [
                {
                    "role": "system",
                    "content": "You are a grounded assistant."
                },
                {
                    "role": "user",
                    "content": "What is the answer?"
                }
            ]
        }"#;
        let parsed = parse_chat_completion(json.as_bytes(), "default-model").unwrap();
        assert!(parsed.is_structured_json);
        assert_eq!(parsed.options.max_new_tokens, 128);
        assert_eq!(parsed.options.sampling.temperature, 0.0);
        assert_eq!(parsed.messages.len(), 2);
        assert_eq!(parsed.messages[0].role, "system");
        assert!(
            parsed.messages[0]
                .text()
                .contains("You are a grounded assistant.")
        );
        assert!(
            parsed.messages[0]
                .text()
                .contains("[RESPONSE FORMAT INSTRUCTION]")
        );
        assert!(parsed.messages[0].text().contains("selected_chunk_ids"));
    }

    #[test]
    fn test_clean_structured_output() {
        // Markdown fence with json
        let raw1 = "```json\n{\n  \"answerable\": true,\n  \"answer\": \"hello\"\n}\n```";
        assert_eq!(
            clean_structured_output(raw1),
            "{\n  \"answerable\": true,\n  \"answer\": \"hello\"\n}"
        );

        // Markdown fence without json
        let raw2 = "```\n{\"answerable\": false}\n```";
        assert_eq!(clean_structured_output(raw2), "{\"answerable\": false}");

        // Already clean json
        let raw3 = "{\"answerable\": true}";
        assert_eq!(clean_structured_output(raw3), "{\"answerable\": true}");

        // Extra conversational fluff around json
        let raw4 = "Here is the response:\n{\"answerable\": true}\nHope this helps!";
        assert_eq!(clean_structured_output(raw4), "{\"answerable\": true}");
    }

    #[test]
    fn test_sse_chunk_formatters() {
        let created = 1720000000;
        let role_chunk = format_sse_chat_role_chunk("chatcmpl-test", "test-model", created);
        assert!(role_chunk.starts_with("data: "));
        assert!(role_chunk.ends_with("\n\n"));
        assert!(role_chunk.contains("\"role\":\"assistant\""));
        assert!(role_chunk.contains("\"content\":\"\""));

        let content_chunk =
            format_sse_chat_content_chunk("chatcmpl-test", "test-model", created, "Xin chào");
        assert!(content_chunk.starts_with("data: "));
        assert!(content_chunk.contains("\"content\":\"Xin chào\""));

        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        };
        let finish_chunk =
            format_sse_chat_finish_chunk("chatcmpl-test", "test-model", created, "stop", usage);
        assert!(finish_chunk.contains("\"finish_reason\":\"stop\""));
        assert!(finish_chunk.contains("\"completion_tokens\":5"));

        let text_content = format_sse_text_content_chunk("cmpl-test", "test-model", created, "Hello");
        assert!(text_content.contains("\"text\":\"Hello\""));
    }

    #[test]
    fn test_stop_matching_helpers() {
        let stop_words = vec!["stop", "end of sequence", "###"];
        assert_eq!(
            check_stop_match("here is the text stop now", &stop_words),
            Some(17)
        );
        assert_eq!(check_stop_match("no matching words here", &stop_words), None);

        // Longest stop prefix testing
        assert_eq!(longest_stop_prefix_len("prefix end of", &stop_words), 6); // "end of" matches prefix of "end of sequence"
        assert_eq!(longest_stop_prefix_len("prefix ###", &stop_words), 3);
        assert_eq!(longest_stop_prefix_len("normal text", &stop_words), 0);
    }
}
