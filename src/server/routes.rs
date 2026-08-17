use serde::{Deserialize, Serialize};

use crate::{
    engine::{GenerationMetrics, GenerationOptions, ServingCompletion},
    generation::{DEFAULT_SAMPLING_SEED, SamplingConfig},
    model::DecodeProfileReport,
};

pub(crate) struct RouteResponse {
    pub status: &'static str,
    pub body: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CompletionRequest {
    prompt: String,
    max_new_tokens: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    repetition_penalty: Option<f32>,
    seed: Option<u64>,
}

#[derive(Serialize)]
struct CompletionResponse {
    object: &'static str,
    choices: Vec<Choice>,
    usage: Usage,
    metrics: GenerationMetrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    profile: Option<DecodeProfileReport>,
}

#[derive(Serialize)]
struct Choice {
    text: String,
    index: usize,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct ErrorResponse<'a> {
    error: &'a str,
}

pub(crate) fn health() -> RouteResponse {
    json_response("200 OK", &serde_json::json!({ "status": "ok" }))
}

pub(crate) fn parse_completion(body: &[u8]) -> Result<(String, GenerationOptions), RouteResponse> {
    let request: CompletionRequest = match serde_json::from_slice(body) {
        Ok(request) => request,
        Err(error) => {
            return Err(error_response(
                "400 Bad Request",
                &format!("invalid JSON: {error}"),
            ));
        }
    };
    let options = match options_from_request(&request) {
        Ok(options) => options,
        Err(error) => return Err(error_response("400 Bad Request", &error)),
    };
    Ok((request.prompt, options))
}

pub(crate) fn completion_response(completion: ServingCompletion, text: String) -> RouteResponse {
    let completion_tokens = completion.token_ids.len();
    json_response(
        "200 OK",
        &CompletionResponse {
            object: "text_completion",
            choices: vec![Choice {
                text,
                index: 0,
                finish_reason: completion.finish_reason,
            }],
            usage: Usage {
                prompt_tokens: completion.prompt_tokens,
                completion_tokens,
                total_tokens: completion.prompt_tokens + completion_tokens,
            },
            metrics: completion.metrics,
            profile: None,
        },
    )
}

fn options_from_request(request: &CompletionRequest) -> Result<GenerationOptions, String> {
    let sampling = SamplingConfig {
        temperature: request.temperature.unwrap_or(0.0),
        top_k: request.top_k.unwrap_or(50),
        repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
        seed: request.seed.unwrap_or(DEFAULT_SAMPLING_SEED),
    };
    sampling.validate().map_err(|error| error.to_string())?;
    let max_new_tokens = request.max_new_tokens.unwrap_or(64);
    if max_new_tokens == 0 {
        return Err("max_new_tokens must be positive".to_string());
    }
    Ok(GenerationOptions {
        max_new_tokens,
        sampling,
    })
}

fn json_response(status: &'static str, value: &impl Serialize) -> RouteResponse {
    match serde_json::to_string(value) {
        Ok(body) => RouteResponse { status, body },
        Err(error) => error_response(
            "500 Internal Server Error",
            &format!("JSON encode failed: {error}"),
        ),
    }
}

pub(crate) fn error_response(status: &'static str, message: &str) -> RouteResponse {
    let body = serde_json::to_string(&ErrorResponse { error: message })
        .unwrap_or_else(|_| "{\"error\":\"response encoding failed\"}".to_string());
    RouteResponse { status, body }
}
