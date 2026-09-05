mod routes;

use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use anyhow::{Context as _, Result, ensure};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::oneshot,
};

use crate::{
    engine::{Engine, PreparedRequest, ServingHandle},
    scheduler::HardwareCostModel,
    tokenizer::Lfm2Tokenizer,
};

const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 1024 * 1024;

pub fn serve(engine: Engine, address: &str, cost_model: HardwareCostModel) -> Result<()> {
    let config = engine.continuous_config(cost_model)?;
    let tokenizer = engine.tokenizer_clone();
    let (handle, receiver) = ServingHandle::channel(config.queue_capacity);
    let address = address.to_string();
    let (ready_sender, ready_receiver) = std::sync::mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name("llm-async-frontend".to_string())
        .spawn(move || {
            if ready_receiver.recv().is_err() {
                eprintln!("GPU owner stopped during initialization");
                return;
            }
            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(error) => {
                    eprintln!("failed to start Tokio runtime: {error}");
                    return;
                }
            };
            if let Err(error) = runtime.block_on(serve_frontend(&address, tokenizer, handle)) {
                eprintln!("async frontend stopped: {error:#}");
            }
        })
        .context("failed to spawn async frontend")?;
    engine
        .run_continuous_owner_radix(config, receiver, ready_sender)
        .map(|_| ())
}

async fn serve_frontend(
    address: &str,
    tokenizer: Lfm2Tokenizer,
    handle: ServingHandle,
) -> Result<()> {
    let listener = TcpListener::bind(address)
        .await
        .with_context(|| format!("failed to bind HTTP server to {address}"))?;
    eprintln!("listening on http://{address} (Tokio + continuous GPU batching)");
    let tokenizer = Arc::new(tokenizer);
    let request_ids = Arc::new(AtomicU64::new(1));
    loop {
        let (stream, _) = listener
            .accept()
            .await
            .context("failed to accept HTTP connection")?;
        let tokenizer = Arc::clone(&tokenizer);
        let handle = handle.clone();
        let request_ids = Arc::clone(&request_ids);
        tokio::spawn(async move {
            if let Err(error) = handle_connection(stream, tokenizer, handle, request_ids).await {
                eprintln!("HTTP request failed: {error:#}");
            }
        });
    }
}

struct GenerationTask {
    token_ids: Vec<u32>,
    options: crate::engine::GenerationOptions,
    model: String,
    stop: Option<routes::StopCondition>,
    stream: bool,
    is_chat: bool,
    is_structured_json: bool,
}

async fn handle_connection(
    mut stream: TcpStream,
    tokenizer: Arc<Lfm2Tokenizer>,
    handle: ServingHandle,
    request_ids: Arc<AtomicU64>,
) -> Result<()> {
    let request_started = Instant::now();
    let (method, path, body) = read_request(&mut stream).await?;
    let default_model = routes::DEFAULT_MODEL_NAME;
    let normalized = path.trim_end_matches('/');
    let normalized_path = if normalized.is_empty() {
        "/"
    } else {
        normalized
    };

    let response = if method == "OPTIONS" {
        routes::cors_preflight()
    } else {
        match (method.as_str(), normalized_path) {
            ("GET", "/health") | ("GET", "/v1/health") => routes::health(),
            ("GET", "/version") | ("GET", "/v1/version") => routes::version(),
            ("GET", "/v1/models") | ("GET", "/models") => routes::models_list(default_model),
            ("GET", p) if p.starts_with("/v1/models/") => {
                let id = &p["/v1/models/".len()..];
                routes::model_retrieve(default_model, id)
            }
            ("GET", p) if p.starts_with("/models/") => {
                let id = &p["/models/".len()..];
                routes::model_retrieve(default_model, id)
            }
            ("POST", "/v1/chat/completions") | ("POST", "/chat/completions") => {
                let parsed = match routes::parse_chat_completion(&body, default_model) {
                    Ok(val) => val,
                    Err(resp) => return write_response(&mut stream, resp).await,
                };
                let tokenization_started = Instant::now();
                let tokenizer_worker = Arc::clone(&tokenizer);
                let messages = parsed.messages;
                let token_ids = match tokio::task::spawn_blocking(move || {
                    tokenizer_worker.encode_chat_prompt(&messages)
                })
                .await
                .context("tokenizer worker panicked")?
                {
                    Ok(tokens) => tokens,
                    Err(error) => {
                        return write_response(
                            &mut stream,
                            routes::error_response(
                                "400 Bad Request",
                                &format!("tokenization failed: {error:#}"),
                                "invalid_request_error",
                                400,
                                Some("messages"),
                            ),
                        )
                        .await;
                    }
                };
                let tokenization_ms = tokenization_started.elapsed().as_secs_f64() * 1000.0;
                let task = GenerationTask {
                    token_ids,
                    options: parsed.options,
                    model: parsed.model,
                    stop: parsed.stop,
                    stream: parsed.stream,
                    is_chat: true,
                    is_structured_json: parsed.is_structured_json,
                };
                execute_generation_task(
                    task,
                    handle,
                    tokenizer,
                    request_ids,
                    request_started,
                    tokenization_ms,
                )
                .await
            }
            ("POST", "/v1/completions") | ("POST", "/completions") => {
                let parsed = match routes::parse_completion(&body, default_model) {
                    Ok(val) => val,
                    Err(resp) => return write_response(&mut stream, resp).await,
                };
                let tokenization_started = Instant::now();
                let tokenizer_worker = Arc::clone(&tokenizer);
                let prompt = parsed.prompt;
                let token_ids = match tokio::task::spawn_blocking(move || {
                    tokenizer_worker.encode_text(&prompt)
                })
                .await
                .context("tokenizer worker panicked")?
                {
                    Ok(tokens) => tokens,
                    Err(error) => {
                        return write_response(
                            &mut stream,
                            routes::error_response(
                                "400 Bad Request",
                                &format!("tokenization failed: {error:#}"),
                                "invalid_request_error",
                                400,
                                Some("prompt"),
                            ),
                        )
                        .await;
                    }
                };
                let tokenization_ms = tokenization_started.elapsed().as_secs_f64() * 1000.0;
                let task = GenerationTask {
                    token_ids,
                    options: parsed.options,
                    model: parsed.model,
                    stop: parsed.stop,
                    stream: parsed.stream,
                    is_chat: false,
                    is_structured_json: false,
                };
                execute_generation_task(
                    task,
                    handle,
                    tokenizer,
                    request_ids,
                    request_started,
                    tokenization_ms,
                )
                .await
            }
            _ => routes::error_response(
                "404 Not Found",
                "route not found",
                "not_found_error",
                404,
                None,
            ),
        }
    };
    write_response(&mut stream, response).await
}

async fn execute_generation_task(
    task: GenerationTask,
    handle: ServingHandle,
    tokenizer: Arc<Lfm2Tokenizer>,
    request_ids: Arc<AtomicU64>,
    request_started: Instant,
    tokenization_ms: f64,
) -> routes::RouteResponse {
    let (response_sender, response_receiver) = oneshot::channel();
    let request_id = request_ids.fetch_add(1, Ordering::Relaxed);
    let prepared = PreparedRequest {
        request_id,
        token_ids: task.token_ids,
        maximum_new_tokens: task.options.max_new_tokens,
        stop_on_eos: true,
        sampling: task.options.sampling,
        arrived: request_started,
        response: response_sender,
    };
    if handle.try_submit(prepared).is_err() {
        return routes::error_response(
            "503 Service Unavailable",
            "engine admission queue is full",
            "server_error",
            503,
            None,
        );
    }
    match response_receiver.await {
        Ok(Ok(mut completion)) => {
            completion.metrics.tokenization_ms = tokenization_ms;
            let decode_started = Instant::now();
            let tokenizer_worker = Arc::clone(&tokenizer);
            let ids = completion.token_ids.clone();
            let text =
                match tokio::task::spawn_blocking(move || tokenizer_worker.decode(&ids)).await {
                    Ok(Ok(decoded)) => decoded,
                    Ok(Err(err)) => {
                        return routes::error_response(
                            "500 Internal Server Error",
                            &format!("detokenization failed: {err:#}"),
                            "internal_error",
                            500,
                            None,
                        );
                    }
                    Err(err) => {
                        return routes::error_response(
                            "500 Internal Server Error",
                            &format!("detokenizer worker panicked: {err:#}"),
                            "internal_error",
                            500,
                            None,
                        );
                    }
                };
            completion.metrics.detokenization_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
            completion.metrics.total_ms = request_started.elapsed().as_secs_f64() * 1000.0;
            if task.is_chat {
                if task.stream {
                    routes::chat_completion_sse_stream(
                        request_id,
                        &task.model,
                        &completion,
                        &text,
                        task.stop.as_ref(),
                        task.is_structured_json,
                    )
                } else {
                    routes::chat_completion_response(
                        request_id,
                        task.model,
                        completion,
                        text,
                        task.stop.as_ref(),
                        task.is_structured_json,
                    )
                }
            } else if task.stream {
                routes::completion_sse_stream(
                    request_id,
                    &task.model,
                    &completion,
                    &text,
                    task.stop.as_ref(),
                )
            } else {
                routes::completion_response(
                    request_id,
                    task.model,
                    completion,
                    text,
                    task.stop.as_ref(),
                )
            }
        }
        Ok(Err(error)) => {
            let (code, err_type) = if error.status.starts_with("400") {
                (400, "invalid_request_error")
            } else if error.status.starts_with("503") {
                (503, "server_error")
            } else {
                (500, "internal_error")
            };
            routes::error_response(error.status, &error.message, err_type, code, None)
        }
        Err(_) => routes::error_response(
            "503 Service Unavailable",
            "GPU owner stopped",
            "server_error",
            503,
            None,
        ),
    }
}

async fn read_request(stream: &mut TcpStream) -> Result<(String, String, Vec<u8>)> {
    let mut buffer = Vec::with_capacity(4096);
    let header_end;
    loop {
        ensure!(
            buffer.len() < MAX_HEADER_BYTES,
            "HTTP headers are too large"
        );
        let mut chunk = [0u8; 4096];
        let read = stream
            .read(&mut chunk)
            .await
            .context("failed to read HTTP request")?;
        ensure!(read > 0, "connection closed before HTTP headers");
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(index) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            header_end = index + 4;
            break;
        }
    }
    let headers =
        std::str::from_utf8(&buffer[..header_end]).context("HTTP headers are not UTF-8")?;
    let mut lines = headers.split("\r\n");
    let request_line = lines.next().context("missing HTTP request line")?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().context("missing HTTP method")?.to_string();
    let path = parts.next().context("missing HTTP path")?.to_string();
    let version = parts.next().context("missing HTTP version")?;
    ensure!(
        version == "HTTP/1.1" || version == "HTTP/1.0",
        "unsupported HTTP version"
    );
    let mut content_length = 0usize;
    for line in lines {
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            content_length = value.trim().parse().context("invalid Content-Length")?;
        }
    }
    ensure!(content_length <= MAX_BODY_BYTES, "HTTP body is too large");
    let total = header_end
        .checked_add(content_length)
        .context("HTTP request size overflow")?;
    while buffer.len() < total {
        let read = stream
            .read_buf(&mut buffer)
            .await
            .context("failed to read HTTP body")?;
        ensure!(read > 0, "connection closed before HTTP body");
    }
    Ok((method, path, buffer[header_end..total].to_vec()))
}

async fn write_response(stream: &mut TcpStream, response: routes::RouteResponse) -> Result<()> {
    let headers = if response.status.starts_with("204") {
        format!(
            "HTTP/1.1 {}\r\nContent-Length: 0\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Max-Age: 86400\r\nConnection: close\r\n\r\n",
            response.status
        )
    } else {
        format!(
            "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\nConnection: close\r\n\r\n",
            response.status,
            response.content_type,
            response.body.len(),
        )
    };
    stream
        .write_all(headers.as_bytes())
        .await
        .context("failed to write HTTP headers")?;
    if !response.body.is_empty() {
        stream
            .write_all(response.body.as_bytes())
            .await
            .context("failed to write HTTP body")?;
    }
    stream
        .shutdown()
        .await
        .context("failed to close HTTP response")
}
