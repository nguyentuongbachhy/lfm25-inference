# lfm25-inference

A from-scratch Rust + CUDA inference runtime and OpenAI-compatible serving engine for `LFM2.5-1.2B-Instruct`.

The project focuses on explicit GPU memory management, custom CUDA kernels,
cuBLASLt GEMMs, paged KV caching, fused activations, low-latency decode, continuous GPU
batching, and an OpenAI-compatible HTTP API server without PyTorch, Candle, GGUF/GGML, ONNX, or Burn.

Key runtime features:
- **OpenAI & vLLM Compatible Server**: Full `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, CORS `OPTIONS`, and Server-Sent Events (SSE) streaming.
- **Structured Outputs (JSON Schema)**: Strict OpenAI `response_format: {"type": "json_schema" | "json_object"}` alongside Ollama `format` / `options` schema compatibility with automated markdown fence stripping.
- **Fused CUDA Kernels**: Fused Residual RMSNorm $\to$ FP8 E4M3 (Sprint 1 Champion) and Fused SwiGLU $\to$ FP8 E4M3 (Sprint 2 Champion) eliminating redundant DRAM roundtrips.
- **Tensor Core FlashAttention**: Contiguous & segmented prefill FlashAttention delivering up to 4.6x prefill speedup.
- **Continuous GPU Serving**: Paged KV cache (PS16) with radix tree state reuse, fused argmax sampling, and 170–180 tokens/sec continuous decode throughput on laptop GPUs.

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

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/v1/chat/completions` | Chat completions with ChatML formatting, streaming SSE, and structured outputs |
| `POST` | `/v1/completions` | Classic text completion API (OpenAI compatible) |
| `GET` | `/v1/models` | List available models (`object: "list"`, `max_model_len: 32768`) |
| `GET` | `/v1/models/{model}` | Retrieve specific model metadata |
| `GET` | `/health` | Server health check (`{"status": "ok"}`) |
| `GET` | `/version` | Runtime version information |
| `OPTIONS`| `*` | CORS preflight with full permissive headers for web UI / browser access |

### Starting the Server

Start the production continuous server on port `8086` (or any custom port):

```bash
LLM_CUDA_ARCH=compute_120 cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --serve 127.0.0.1:8086 \
  --hardware-profile docs/serving/fp8-splitk-hardware-ps16.cost-model.json \
  --page-size 16
```

> [!NOTE]
> The selective FP8 policy (`docs/benchmarks/fp8/selected-policy.json`) is auto-detected at startup if present. To force BF16-only serving, omit the FP8 policy and hardware profile.

### API Usage Examples

#### 1. Chat Completion (Non-Streaming)

```bash
curl -s -X POST http://127.0.0.1:8086/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LFM2.5-1.2B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "Explain quantum computing in one sentence."}
    ],
    "max_tokens": 64,
    "temperature": 0.0
  }'
```

#### 2. Streaming Chat Completion (SSE)

```bash
curl -N -s -X POST http://127.0.0.1:8086/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LFM2.5-1.2B-Instruct",
    "messages": [
      {"role": "user", "content": "Count from 1 to 5."}
    ],
    "max_tokens": 64,
    "stream": true
  }'
```

#### 3. Structured Output (OpenAI `json_schema`)

Enforce strict JSON schema validation for reliable tool calling, information extraction, and agent workflows:

```bash
curl -s -X POST http://127.0.0.1:8086/v1/chat/completions \
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
curl -s -X POST http://127.0.0.1:8086/v1/chat/completions \
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
curl -s -X POST http://127.0.0.1:8086/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Liquid Foundation Models are",
    "max_tokens": 32,
    "temperature": 0.0
  }'
```

#### 6. CORS Preflight Check

```bash
curl -s -I -X OPTIONS http://127.0.0.1:8086/v1/chat/completions
# Returns HTTP/1.1 204 No Content with Access-Control-Allow-Origin: *
```

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