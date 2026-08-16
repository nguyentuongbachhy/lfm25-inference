use serde::{Deserialize, Serialize};

use crate::{
    engine::{GenerationMetrics, GenerationOptions, ServingCompletion},
    generation::SamplingConfig,
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

pub(crate) fn completion_response(
    completion: ServingCompletion,
    text: String,
    completion_tokens: usize,
) -> RouteResponse {
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
        seed: request.seed.unwrap_or(0x4c_46_4d_32),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_options_validate_sampling() {
        let request = CompletionRequest {
            prompt: "hello".to_string(),
            max_new_tokens: Some(8),
            temperature: Some(0.2),
            top_k: Some(20),
            repetition_penalty: Some(1.05),
            seed: Some(7),
        };
        let options = options_from_request(&request).expect("valid options");
        assert_eq!(options.max_new_tokens, 8);
        assert_eq!(options.sampling.top_k, 20);
    }

    #[test]
    fn request_rejects_engine_page_size_override() {
        let body = br#"{"prompt":"hello","page_size":32}"#;
        let error = match serde_json::from_slice::<CompletionRequest>(body) {
            Ok(_) => panic!("page_size must remain an engine-level setting"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("unknown field `page_size`"));
    }
}
