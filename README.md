# lfm25-inference

A from-scratch Rust + CUDA inference runtime and OpenAI-compatible serving engine for `LFM2.5-1.2B-Instruct`.

The project focuses on explicit GPU memory management, custom CUDA kernels,
cuBLASLt GEMMs, paged KV caching, fused activations, low-latency decode, continuous GPU
batching, and an OpenAI-compatible HTTP API server without PyTorch, Candle, GGUF/GGML, ONNX, or Burn.

Key runtime features:
- **OpenAI & Ollama Drop-in Serving**: Full OpenAI `/v1/chat/completions`, `/v1/completions`, `/v1/models` and native Ollama `/api/tags`, `/api/version`, `/api/show`, `/api/chat`, `/api/generate` endpoints with NDJSON streaming.
- **High Concurrency Continuous Batching**: Zero-overhead continuous batching scaling from 151 tok/s ($C=1$) to **1,006.8 tok/s** ($C=8$) on an RTX 5060 Laptop GPU.
- **Multi-Turn Radix Tree Prefix Caching**: Automated KV block reuse across conversation turns delivering **6.15x TTFT speedup** on multi-turn chats.
- **CUDA Graphs Decode Acceleration**: Default-enabled lazy graph capture promoting single-stream decode host launch overhead down by 9.3x.
- **Structured Outputs (JSON Schema)**: Strict OpenAI `response_format: {"type": "json_schema" | "json_object"}` alongside Ollama `format` / `options` schema compatibility with automated markdown fence stripping.
- **Fused CUDA Kernels**: Fused Residual RMSNorm $\to$ FP8 E4M3 (Sprint 1 Champion) and Fused SwiGLU $\to$ FP8 E4M3 (Sprint 2 Champion) eliminating redundant DRAM roundtrips.
- **Tensor Core FlashAttention**: Contiguous & segmented prefill FlashAttention delivering up to 4.6x prefill speedup.

The current validated target is an NVIDIA GeForce RTX 5060 Laptop GPU
(Blackwell GeForce SM120) with CUDA 12.8.x. BF16 remains the golden reference
and fallback path. A checkpoint- and GPU-specific selective E4M3 policy is
available for decode.

For the complete validated results, see
[`docs/final_release_report.md`](docs/final_release_report.md) and [`walkthrough.md`](walkthrough.md).

## Requirements

Recommended environment for reproducing the validated runtime:

- Linux or WSL2
- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.8.x for the validated SM120 build
- Rust toolchain
- `LFM2.5-1.2B-Instruct` checkpoint files

Verify the GPU and CUDA compiler first:

```bash
nvidia-smi
nvcc --version
```

The measured build uses:

```text
GPU architecture: compute_120
CUDA compiler:    12.8.x
```

## Repository Setup

Clone the repository:

```bash
git clone https://github.com/nguyentuongbachhy/lfm25-inference.git
cd lfm25-inference
```

Place the model under:

```text
models/LFM2.5-1.2B-Instruct/
```

The runtime expects the Hugging Face model/tokenizer/config files in that
directory, including the model safetensors checkpoint.

## Build

For the validated RTX 5060 Laptop / SM120 target:

```bash
LLM_CUDA_ARCH=compute_120 cargo build --release
```

Check the project:

```bash
LLM_CUDA_ARCH=compute_120 cargo check --all-features
```

Run the correctness suite:

```bash
LLM_CUDA_ARCH=compute_120 cargo test --release -- --test-threads=1
```

GPU tests and benchmarks require direct CUDA device access.

## Run Inference

### BF16 reference path

BF16 is the default path and should be used as the correctness reference:

```bash
LLM_CUDA_ARCH=compute_120 cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --prompt "Who are you?" \
  --max-new-tokens 64 \
  --page-size 16 \
  --temperature 0.0
```

### Selective E4M3 decode

The validated policy keeps prefill, KV cache, attention math, normalization,
RoPE, residual operations, embeddings, and sampling outside the selective FP8
GEMM path.

Enable the frozen decode policy explicitly at startup:

```bash
LLM_CUDA_ARCH=compute_120 cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --fp8-policy docs/benchmarks/fp8/selected-policy.json \
  --prompt "Who are you?" \
  --max-new-tokens 128 \
  --page-size 16 \
  --temperature 0.0
```

The promoted policy is specific to the measured checkpoint and GPU. Re-run
calibration, held-out validation, and performance benchmarks before treating it
as validated on a different checkpoint or GPU.

### Sampling

Example stochastic generation:

```bash
LLM_CUDA_ARCH=compute_120 cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --prompt "Write one sentence about CUDA." \
  --max-new-tokens 64 \
  --temperature 0.1 \
  --top-k 50 \
  --repetition-penalty 1.05 \
  --seed 42
```

## Serving

The runtime includes an OpenAI-compatible and vLLM-aligned HTTP API server powered by high-performance continuous GPU batching, paged KV caching, and fused kernels.

### Endpoints Overview

| Method | Endpoint | Protocol | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/v1/chat/completions` | OpenAI | Chat completions with ChatML formatting, SSE streaming, and structured outputs |
| `POST` | `/v1/completions` | OpenAI | Classic text completion API (OpenAI compatible) |
| `GET` | `/v1/models` | OpenAI | List available models (`object: "list"`, `max_model_len: 32768`) |
| `GET` | `/v1/models/{model}` | OpenAI | Retrieve specific model metadata |
| `POST` | `/api/chat` | Ollama | Native Ollama chat endpoint with NDJSON streaming (`stream: true` default) |
| `POST` | `/api/generate` | Ollama | Native Ollama text completion endpoint with NDJSON streaming |
| `GET` | `/api/tags` | Ollama | List models for OpenWebUI / Ollama CLI discovery |
| `POST` | `/api/show` | Ollama | Show model architecture, context length, template, and parameters |
| `GET` | `/api/version` | Ollama | Ollama version compatibility probe (`{"version": "0.1.0"}`) |
| `GET` | `/health` | Generic | Server health check (`{"status": "ok"}`) |
| `GET` | `/version` | Generic | Runtime version information |
| `OPTIONS`| `*` | CORS | CORS preflight with full permissive headers for web UI / browser access |

### Starting the Server

Start the production continuous server on port `8088` (or any custom port):

```bash
LLM_CUDA_ARCH=compute_120 cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --serve 127.0.0.1:8088 \
  --hardware-profile docs/serving/fp8-splitk-hardware-ps16.cost-model.json
```

> [!NOTE]
> The selective FP8 policy (`docs/benchmarks/fp8/selected-policy.json`) is auto-detected at startup if present. If your environment uses an HTTP proxy, specify `--noproxy "*"` with `curl` to ensure direct loopback requests.

### Automated End-to-End Evaluation & Verification

To verify that all endpoints (OpenAI & Ollama) work correctly and reproduce the benchmark numbers with a single command:

```bash
./scripts/run_serving_evaluation.sh 8088
```

This master script automatically:
1. Compiles the release binary (if needed).
2. Spawns the server on `127.0.0.1:8088`.
3. Verifies all OpenAI (`/v1/*`) and Ollama (`/api/*`) endpoints with `scripts/serving/test_all_endpoints.py`.
4. Executes the Multi-Turn Radix Tree Prefix Caching benchmark with `scripts/serving/bench_prefix_caching.py`.
5. Executes the Continuous Batching Concurrency benchmark ($C=1, 2, 4, 8$) with `scripts/serving/bench_concurrency.py`.
6. Gracefully shuts down the background server upon completion.

### API Usage Examples

#### 1. Chat Completion (Non-Streaming)

```bash
curl --noproxy "*" -s -X POST http://127.0.0.1:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LFM2.5-1.2B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "Explain quantum computing in one sentence."}
    ],
    "max_tokens": 100,
    "temperature": 0.0
  }'
```

#### 2. Streaming Chat Completion (OpenAI SSE)

```bash
curl --noproxy "*" -N -s -X POST http://127.0.0.1:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LFM2.5-1.2B-Instruct",
    "messages": [
      {"role": "user", "content": "Count from 1 to 5."}
    ],
    "max_tokens": 100,
    "stream": true
  }'
```

#### 3. Structured Output (OpenAI `json_schema`)

Enforce strict JSON schema validation for reliable tool calling, information extraction, and agent workflows:

```bash
curl --noproxy "*" -s -X POST http://127.0.0.1:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LFM2.5-1.2B-Instruct",
    "messages": [
      {"role": "user", "content": "List 2 planets and their order from the sun."}
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "planets_list",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "planets": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "name": {"type": "string"},
                  "order": {"type": "integer"}
                },
                "required": ["name", "order"]
              }
            }
          },
          "required": ["planets"]
        }
      }
    },
    "max_tokens": 128
  }'
```

#### 4. Ollama Payload Compatibility (`format` + `options`)

Clients configured for Ollama can call `/v1/chat/completions` directly without modification:

```bash
curl --noproxy "*" -s -X POST http://127.0.0.1:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
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
        "content": "You are a grounded enterprise assistant. Answer only from the evidence."
      },
      {
        "role": "user",
        "content": "Evidence: [chunk_01] Apollo 11 landed in 1969.\nQuestion: When did Apollo 11 land?"
      }
    ]
  }'
```

The server automatically injects the schema instruction, strips markdown code blocks, and returns a clean, parseable JSON object in `choices[0].message.content`.

#### 5. Text Completion (`/v1/completions`)

```bash
curl --noproxy "*" -s -X POST http://127.0.0.1:8088/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Liquid Foundation Models are",
    "max_tokens": 64,
    "temperature": 0.0
  }'
```

#### 6. Native Ollama Chat (`/api/chat` - NDJSON Streaming)

Drop-in replacement for Ollama clients and OpenWebUI. By default, `stream` is `true` emitting newline-delimited JSON (NDJSON) chunks:

```bash
curl --noproxy "*" -N -s -X POST http://127.0.0.1:8088/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LFM2.5-1.2B-Instruct",
    "messages": [
      {"role": "user", "content": "Explain photosynthesis in 2 sentences."}
    ],
    "options": {
      "num_predict": 100,
      "temperature": 0.0
    }
  }'
```

#### 7. Native Ollama Raw Generation (`/api/generate`)

```bash
curl --noproxy "*" -N -s -X POST http://127.0.0.1:8088/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LFM2.5-1.2B-Instruct",
    "prompt": "The capital of France is",
    "options": {
      "num_predict": 30,
      "temperature": 0.0
    }
  }'
```

#### 8. CORS Preflight Check

```bash
curl --noproxy "*" -s -I -X OPTIONS http://127.0.0.1:8088/v1/chat/completions
# Returns HTTP/1.1 204 No Content with Access-Control-Allow-Origin: *
```

### Performance & Scalability Evidence

#### Continuous Batching Concurrency Scaling (RTX 5060 Laptop GPU)

Continuous serving throughput under simultaneous active client streams ($C \in \{1, 2, 4, 8\}$):

| Concurrency ($C$) | Total Tokens | Wall Clock (s) | Aggregate Throughput | Per-Stream Throughput | Mean TTFT | Scaling Efficiency |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 44 tok | 0.29 s | **151.1 tok/s** | 151.1 tok/s | 18.14 ms | 1.00x |
| **2** | 83 tok | 0.31 s | **270.0 tok/s** | 135.0 tok/s | 20.46 ms | 1.79x |
| **4** | 170 tok | 0.32 s | **530.9 tok/s** | 132.7 tok/s | 29.93 ms | 3.51x |
| **8** | 333 tok | 0.33 s | **1,006.8 tok/s** | 125.9 tok/s | 34.10 ms | **6.66x (>1k tok/s)** |

#### Multi-Turn Radix Tree Prefix Caching Speedup

Measured on a ~600-token enterprise document context:

| Turn | Status | Prompt Size | Time-to-First-Token (TTFT) | Total Time | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Turn 1** | Cold Prefill | ~600 tokens | **170.47 ms** | 291.53 ms | Baseline (1.0x) |
| **Turn 2** | Radix Cache Hit | ~630 tokens (prefix cached) | **27.74 ms** | 125.90 ms | **6.15x Faster TTFT** |

`--page-size` and precision selection are engine startup policies. They are not
request-level overrides.

## Profiling

Use coarse profiling first:

```bash
LLM_CUDA_ARCH=compute_120 cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --prompt "Write the integers from 1 to 1000, separated by commas." \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --profile-decode coarse \
  --profile-warmup-steps 8 \
  --profile-steps 100
```

Use detailed profiling only for bottleneck attribution after coarse profiling
identifies the dominant region. Detailed CUDA-event instrumentation adds
measurement overhead and should not be used as the uninstrumented latency
baseline.

## Benchmarking

Paged decode attention:

```bash
LLM_CUDA_ARCH=compute_120 cargo test --release \
  bench_paged_attention_lfm2_bf16 -- \
  --ignored --nocapture --test-threads=1
```

Contiguous prefill attention:

```bash
LLM_CUDA_ARCH=compute_120 cargo test --release \
  bench_prefill_attention_lfm2_bf16 -- \
  --ignored --nocapture --test-threads=1
```

Serving decode benchmark:

```bash
LLM_CUDA_ARCH=compute_120 cargo run --release -- \
  --benchmark-serving docs/serving/ps16-decode.json \
  --page-size 16
```

Load/goodput benchmark using the validated selective-E4M3 scheduler profile:

```bash
LLM_CUDA_ARCH=compute_120 cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --fp8-policy docs/benchmarks/fp8/selected-policy.json \
  --benchmark-load docs/serving/ps16-load.json \
  --hardware-profile docs/serving/fp8-splitk-hardware-ps16.cost-model.json \
  --page-size 16
```

### Serving & Concurrency Benchmarks

Run the all-in-one automated serving benchmark suite:

```bash
./scripts/run_serving_evaluation.sh 8088
```

Or run dedicated Python benchmark harnesses against an already running server:

```bash
# Verify all OpenAI and Ollama endpoints
python3 scripts/serving/test_all_endpoints.py http://127.0.0.1:8088

# Benchmark Multi-Turn Radix Tree Prefix Caching (TTFT speedup)
python3 scripts/serving/bench_prefix_caching.py http://127.0.0.1:8088

# Benchmark Continuous Batching Concurrency Scaling (C = 1, 2, 4, 8)
python3 scripts/serving/bench_concurrency.py http://127.0.0.1:8088
```

More benchmark and validation commands are collected in
[`docs/command.md`](docs/command.md).

## FP8 Calibration and Validation

Calibration is an offline workflow and always starts from the BF16 reference
model.

Example calibration:

```bash
LLM_CUDA_ARCH=compute_120 cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --calibrate-fp8 path/to/calibration.txt \
  --fp8-eval-corpus path/to/disjoint-validation.txt \
  --calibration-output docs/benchmarks/fp8/calibration-summary.json \
  --calibration-max-sequences 256 \
  --calibration-max-tokens 1024 \
  --fp8-eval-sequences 16 \
  --fp8-eval-max-tokens 128
```

Evaluate a frozen policy on an independent test corpus:

```bash
LLM_CUDA_ARCH=compute_120 cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --evaluate-fp8 docs/benchmarks/fp8/selected-policy.json \
  --fp8-eval-corpus path/to/disjoint-test.txt \
  --fp8-eval-sequences 32 \
  --fp8-eval-max-tokens 256 \
  --evaluation-output docs/benchmarks/final/quality.json
```

The production decode path does not perform calibration, CPU readback, or policy
search.

## Runtime Design

The runtime uses a small set of production principles:

- BF16 remains the reproducible reference and fallback.
- Prefill and decode are treated as different workloads.
- Model weights are resolved into typed structures at startup.
- Decode avoids hot-path string lookup and unnecessary allocation.
- Temporary device memory is reused through bounded pools.
- KV pages are managed by the engine rather than by individual requests.
- Precision and page size are startup policies.
- Numerical optimizations are validated through downstream hidden/logit quality,
  not only local kernel error.
- Performance changes are promoted using end-to-end measurements, not isolated
  microbenchmarks alone.

The current runtime includes:
- **Paged GQA/XQA Decode Attention**: Zero-allocation paged KV caching with Split-K reduction.
- **Tensor Core FlashAttention**: Contiguous and segmented/ragged multi-sequence prefill attention.
- **Fused Residual RMSNorm $\to$ FP8 E4M3**: Sprint 1 Champion eliminating 9 DRAM write/read roundtrips per token with 100.0% bitwise parity.
- **Fused SwiGLU $\to$ FP8 E4M3**: Sprint 2 Champion eliminating 7 DRAM write/read roundtrips across all FP8 down-projection layers with 100.0% bitwise parity.
- **Scratchless Atomic Greedy Argmax**: High-speed reduction bypassing CPU synchronization and intermediate staging.
- **Multilingual Adaptive Speculative Decoding**: DSpark dynamic backoff with 0 auxiliary VRAM.
- **OpenAI & vLLM Serving Stack**: Continuous GPU batching, SSE streaming, and schema-guided structured outputs.

## Project Layout

The main directories are:

```text
src/        Rust runtime, model, engine, serving, and CUDA integration
kernels/    CUDA kernel sources
models/     local model checkpoints (not expected to be committed)
docs/       commands, benchmark evidence, research decisions, and roadmap
research/   isolated experimental implementations and harnesses
scripts/    benchmark and research automation
```

## Documentation

Start with these files:

- [`docs/final_release_report.md`](docs/final_release_report.md) — consolidated
  validated milestone and final results.
- [`docs/command.md`](docs/command.md) — runnable correctness, benchmark,
  profiling, serving, and FP8 commands.
- [`docs/optimization.md`](docs/optimization.md) — historical implemented
  optimizations and measured bottlenecks.
- [`docs/fp8_report.md`](docs/fp8_report.md) — selective E4M3 calibration,
  quality, and E2E evidence.
- [`docs/benchmarks/validated-runtime-v2.md`](docs/benchmarks/validated-runtime-v2.md)
  — validated atomic argmax production delta.
- [`docs/research/nvfp4_rejection.md`](docs/research/nvfp4_rejection.md) — why
  NVFP4 was rejected for production despite strong primitive performance.
- [`docs/architecture/fp8_kv_cache_roadmap.md`](docs/architecture/fp8_kv_cache_roadmap.md) —
  technical architecture and kernel design blueprint for FP8 E4M3 KV cache.
- [`docs/next_optimization_roadmap.md`](docs/next_optimization_roadmap.md) —
  future optimization directions, gates, and stop conditions.

## Current Scope and Limitations

- The runtime is currently specialized for `LFM2.5-1.2B-Instruct`.
- The published performance evidence is specific to the measured RTX 5060
  Laptop GPU / SM120 environment.
- The validated E4M3 policy targets selected M=1 decode GEMMs.
- BF16 weights remain resident alongside persistent selected FP8 copies.
- Prefill remains BF16.
- KV cache remains BF16.
- Long-context decode receives less benefit from weight-only FP8 because
  attention and KV traffic become a larger fraction of TPOT.
- Research branches may contain rejected or experimental code that is not part
  of the production `main` path.

Future work is intentionally kept out of the README. See
[`docs/next_optimization_roadmap.md`](docs/next_optimization_roadmap.md) before
starting another optimization campaign.

## License

No license has been specified yet.