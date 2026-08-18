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
    // From this point onward the Engine is only touched by this dedicated GPU
    // owner thread. Move synchronization-heavy runtime state into owner-local
    // storage before the serving warmup populates its steady-state caches.
    engine.enter_serving_owner_mode()?;
    engine
        .run_continuous_owner(config, receiver, ready_sender)
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

async fn handle_connection(
    mut stream: TcpStream,
    tokenizer: Arc<Lfm2Tokenizer>,
    handle: ServingHandle,
    request_ids: Arc<AtomicU64>,
) -> Result<()> {
    let request_started = Instant::now();
    let (method, path, body) = read_request(&mut stream).await?;
    let response = match (method.as_str(), path.as_str()) {
        ("GET", "/health") => routes::health(),
        ("POST", "/v1/completions") => {
            let (prompt, options) = match routes::parse_completion(&body) {
                Ok(value) => value,
                Err(response) => return write_response(&mut stream, response).await,
            };
            let tokenization_started = Instant::now();
            let tokenizer_worker = Arc::clone(&tokenizer);
            let token_ids = match tokio::task::spawn_blocking(move || {
                tokenizer_worker.encode_user_prompt(&prompt)
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
                        ),
                    )
                    .await;
                }
            };
            let tokenization_ms = tokenization_started.elapsed().as_secs_f64() * 1000.0;
            let (response_sender, response_receiver) = oneshot::channel();
            let prepared = PreparedRequest {
                request_id: request_ids.fetch_add(1, Ordering::Relaxed),
                token_ids,
                maximum_new_tokens: options.max_new_tokens,
                stop_on_eos: true,
                sampling: options.sampling,
                arrived: request_started,
                response: response_sender,
            };
            if handle.try_submit(prepared).is_err() {
                return write_response(
                    &mut stream,
                    routes::error_response(
                        "503 Service Unavailable",
                        "engine admission queue is full",
                    ),
                )
                .await;
            }
            match response_receiver.await {
                Ok(Ok(mut completion)) => {
                    completion.metrics.tokenization_ms = tokenization_ms;
                    let decode_started = Instant::now();
                    let tokenizer_worker = Arc::clone(&tokenizer);
                    let ids = completion.token_ids.clone();
                    let text = tokio::task::spawn_blocking(move || tokenizer_worker.decode(&ids))
                        .await
                        .context("detokenizer worker panicked")??;
                    completion.metrics.detokenization_ms =
                        decode_started.elapsed().as_secs_f64() * 1000.0;
                    completion.metrics.total_ms = request_started.elapsed().as_secs_f64() * 1000.0;
                    routes::completion_response(completion, text)
                }
                Ok(Err(error)) => routes::error_response(error.status, &error.message),
                Err(_) => routes::error_response("503 Service Unavailable", "GPU owner stopped"),
            }
        }
        _ => routes::error_response("404 Not Found", "route not found"),
    };
    write_response(&mut stream, response).await
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
    let headers = format!(
        "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.status,
        response.body.len(),
    );
    stream
        .write_all(headers.as_bytes())
        .await
        .context("failed to write HTTP headers")?;
    stream
        .write_all(response.body.as_bytes())
        .await
        .context("failed to write HTTP body")?;
    stream
        .shutdown()
        .await
        .context("failed to close HTTP response")
}
