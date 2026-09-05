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
    sync::{mpsc, oneshot},
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
                if task.stream && !task.is_structured_json {
                    return execute_live_streaming_task(
                        stream,
                        task,
                        handle,
                        tokenizer,
                        request_ids,
                        request_started,
                        tokenization_ms,
                    )
                    .await;
                }
                let response = execute_generation_task(
                    task,
                    handle,
                    tokenizer,
                    request_ids,
                    request_started,
                    tokenization_ms,
                )
                .await;
                return write_response(&mut stream, response).await;
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
                if task.stream {
                    return execute_live_streaming_task(
                        stream,
                        task,
                        handle,
                        tokenizer,
                        request_ids,
                        request_started,
                        tokenization_ms,
                    )
                    .await;
                }
                let response = execute_generation_task(
                    task,
                    handle,
                    tokenizer,
                    request_ids,
                    request_started,
                    tokenization_ms,
                )
                .await;
                return write_response(&mut stream, response).await;
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

async fn write_http_chunk(stream: &mut TcpStream, data: &str) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    let header = format!("{:x}\r\n", data.len());
    stream
        .write_all(header.as_bytes())
        .await
        .context("failed to write HTTP chunk header")?;
    stream
        .write_all(data.as_bytes())
        .await
        .context("failed to write HTTP chunk payload")?;
    stream
        .write_all(b"\r\n")
        .await
        .context("failed to write HTTP chunk delimiter")?;
    stream
        .flush()
        .await
        .context("failed to flush HTTP chunk")?;
    Ok(())
}

async fn write_http_chunk_end(stream: &mut TcpStream) -> Result<()> {
    stream
        .write_all(b"0\r\n\r\n")
        .await
        .context("failed to write HTTP chunk end")?;
    stream
        .flush()
        .await
        .context("failed to flush HTTP chunk end")?;
    Ok(())
}

async fn execute_live_streaming_task(
    mut stream: TcpStream,
    task: GenerationTask,
    handle: ServingHandle,
    tokenizer: Arc<Lfm2Tokenizer>,
    request_ids: Arc<AtomicU64>,
    request_started: Instant,
    _tokenization_ms: f64,
) -> Result<()> {
    let (response_sender, response_receiver) = oneshot::channel();
    let (token_sender, mut token_receiver) = mpsc::unbounded_channel();
    let request_id = request_ids.fetch_add(1, Ordering::Relaxed);
    let prompt_tokens = task.token_ids.len();
    let prepared = PreparedRequest {
        request_id,
        token_ids: task.token_ids,
        maximum_new_tokens: task.options.max_new_tokens,
        stop_on_eos: true,
        sampling: task.options.sampling,
        arrived: request_started,
        response: response_sender,
        token_stream: Some(token_sender),
    };

    if handle.try_submit(prepared).is_err() {
        let response = routes::error_response(
            "503 Service Unavailable",
            "engine admission queue is full",
            "server_error",
            503,
            None,
        );
        return write_response(&mut stream, response).await;
    }

    let _ = stream.set_nodelay(true);
    let created = routes::current_timestamp();
    let req_id_str = if task.is_chat {
        format!("chatcmpl-{:016x}{:08x}", created, request_id)
    } else {
        format!("cmpl-{:016x}{:08x}", created, request_id)
    };

    let headers = "HTTP/1.1 200 OK\r\n\
        Content-Type: text/event-stream; charset=utf-8\r\n\
        Cache-Control: no-cache\r\n\
        Transfer-Encoding: chunked\r\n\
        Access-Control-Allow-Origin: *\r\n\
        Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
        Access-Control-Allow-Headers: *\r\n\
        Connection: close\r\n\r\n";
    stream
        .write_all(headers.as_bytes())
        .await
        .context("failed to write streaming HTTP headers")?;
    stream
        .flush()
        .await
        .context("failed to flush streaming HTTP headers")?;

    if task.is_chat {
        let role_chunk = routes::format_sse_chat_role_chunk(&req_id_str, &task.model, created);
        write_http_chunk(&mut stream, &role_chunk).await?;
    }

    let stop_words: Vec<String> = task
        .stop
        .as_ref()
        .map(|s| s.as_vec().into_iter().map(|w| w.to_string()).collect())
        .unwrap_or_default();
    let stop_words_refs: Vec<&str> = stop_words.iter().map(|s| s.as_str()).collect();

    let mut accumulated_tokens = Vec::new();
    let mut emitted_text = String::new();
    let mut stopped = false;

    while let Some(token) = token_receiver.recv().await {
        accumulated_tokens.push(token);
        let current_text = match tokenizer.decode(&accumulated_tokens) {
            Ok(t) => t,
            Err(_) => continue,
        };

        if current_text.ends_with('\u{FFFD}') {
            continue;
        }

        if !stop_words_refs.is_empty() {
            if let Some(pos) = routes::check_stop_match(&current_text, &stop_words_refs) {
                let valid = &current_text[..pos];
                let prefix_len = if valid.starts_with(&emitted_text) {
                    emitted_text.len()
                } else {
                    valid
                        .char_indices()
                        .zip(emitted_text.char_indices())
                        .take_while(|((_, c1), (_, c2))| c1 == c2)
                        .last()
                        .map(|((i, c), _)| i + c.len_utf8())
                        .unwrap_or(0)
                };
                if valid.len() > prefix_len {
                    let delta = &valid[prefix_len..];
                    let chunk = if task.is_chat {
                        routes::format_sse_chat_content_chunk(
                            &req_id_str,
                            &task.model,
                            created,
                            delta,
                        )
                    } else {
                        routes::format_sse_text_content_chunk(
                            &req_id_str,
                            &task.model,
                            created,
                            delta,
                        )
                    };
                    let _ = write_http_chunk(&mut stream, &chunk).await;
                    emitted_text.push_str(delta);
                }
                stopped = true;
                break;
            }

            let hold_len = routes::longest_stop_prefix_len(&current_text, &stop_words_refs);
            let safe_len = current_text.len().saturating_sub(hold_len);
            if safe_len > emitted_text.len() {
                let safe_text = &current_text[..safe_len];
                let prefix_len = if safe_text.starts_with(&emitted_text) {
                    emitted_text.len()
                } else {
                    safe_text
                        .char_indices()
                        .zip(emitted_text.char_indices())
                        .take_while(|((_, c1), (_, c2))| c1 == c2)
                        .last()
                        .map(|((i, c), _)| i + c.len_utf8())
                        .unwrap_or(0)
                };
                if safe_text.len() > prefix_len {
                    let delta = &safe_text[prefix_len..];
                    let chunk = if task.is_chat {
                        routes::format_sse_chat_content_chunk(
                            &req_id_str,
                            &task.model,
                            created,
                            delta,
                        )
                    } else {
                        routes::format_sse_text_content_chunk(
                            &req_id_str,
                            &task.model,
                            created,
                            delta,
                        )
                    };
                    if write_http_chunk(&mut stream, &chunk).await.is_err() {
                        stopped = true;
                        break;
                    }
                    emitted_text.push_str(delta);
                }
            }
        } else {
            let prefix_len = if current_text.starts_with(&emitted_text) {
                emitted_text.len()
            } else {
                current_text
                    .char_indices()
                    .zip(emitted_text.char_indices())
                    .take_while(|((_, c1), (_, c2))| c1 == c2)
                    .last()
                    .map(|((i, c), _)| i + c.len_utf8())
                    .unwrap_or(0)
            };
            if current_text.len() > prefix_len {
                let delta = &current_text[prefix_len..];
                let chunk = if task.is_chat {
                    routes::format_sse_chat_content_chunk(&req_id_str, &task.model, created, delta)
                } else {
                    routes::format_sse_text_content_chunk(&req_id_str, &task.model, created, delta)
                };
                if write_http_chunk(&mut stream, &chunk).await.is_err() {
                    stopped = true;
                    break;
                }
                emitted_text.push_str(delta);
            }
        }
    }

    drop(token_receiver);

    let completion = match response_receiver.await {
        Ok(Ok(c)) => c,
        _ => {
            let _ = write_http_chunk_end(&mut stream).await;
            let _ = stream.shutdown().await;
            return Ok(());
        }
    };

    if !stopped {
        let final_text = tokenizer.decode(&accumulated_tokens).unwrap_or_default();
        let mut cleaned = final_text.as_str();
        for special in ["<|im_end|>", "<|endoftext|>"] {
            if let Some(stripped) = cleaned.strip_suffix(special) {
                cleaned = stripped;
            }
        }
        let prefix_len = if cleaned.starts_with(&emitted_text) {
            emitted_text.len()
        } else {
            cleaned
                .char_indices()
                .zip(emitted_text.char_indices())
                .take_while(|((_, c1), (_, c2))| c1 == c2)
                .last()
                .map(|((i, c), _)| i + c.len_utf8())
                .unwrap_or(0)
        };
        if cleaned.len() > prefix_len {
            let delta = &cleaned[prefix_len..];
            let chunk = if task.is_chat {
                routes::format_sse_chat_content_chunk(&req_id_str, &task.model, created, delta)
            } else {
                routes::format_sse_text_content_chunk(&req_id_str, &task.model, created, delta)
            };
            let _ = write_http_chunk(&mut stream, &chunk).await;
        }
    }

    let finish_reason = if stopped {
        "stop"
    } else {
        completion.finish_reason
    };
    let completion_tokens = accumulated_tokens.len();
    let usage = routes::Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
    };
    let finish_chunk = if task.is_chat {
        routes::format_sse_chat_finish_chunk(
            &req_id_str,
            &task.model,
            created,
            finish_reason,
            usage,
        )
    } else {
        routes::format_sse_text_finish_chunk(
            &req_id_str,
            &task.model,
            created,
            finish_reason,
            usage,
        )
    };
    let _ = write_http_chunk(&mut stream, &finish_chunk).await;
    let _ = write_http_chunk(&mut stream, "data: [DONE]\n\n").await;
    let _ = write_http_chunk_end(&mut stream).await;
    let _ = stream.shutdown().await;
    Ok(())
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
        token_stream: None,
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
