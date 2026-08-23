# lfm25-inference

A from-scratch Rust + CUDA inference runtime for `LFM2.5-1.2B-Instruct`.

The project focuses on explicit GPU memory management, custom CUDA kernels,
cuBLASLt GEMMs, paged KV caching, low-latency decode, and measurement-driven
runtime optimization without PyTorch, Candle, GGUF/GGML, ONNX, or Burn.

The current validated target is an NVIDIA GeForce RTX 5060 Laptop GPU
(Blackwell GeForce SM120) with CUDA 12.8.x. BF16 remains the golden reference
and fallback path. A checkpoint- and GPU-specific selective E4M3 policy is
available for decode.

For the complete validated results, see
[`docs/final_release_report.md`](docs/final_release_report.md).

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

Generate a hardware cost model for the selected page size:

```bash
LLM_CUDA_ARCH=compute_120 cargo run --release -- \
  --benchmark-hardware docs/serving/rtx5060-ps16-hardware.json \
  --page-size 16
```

Start the server:

```bash
LLM_CUDA_ARCH=compute_120 cargo run --release -- \
  --serve 127.0.0.1:8080 \
  --hardware-profile docs/serving/rtx5060-ps16-hardware.cost-model.json \
  --page-size 16
```

Health check:

```bash
curl http://127.0.0.1:8080/health
```

Completion request:

```bash
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Who are you?","max_new_tokens":32,"temperature":0.0}'
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

Load/goodput benchmark:

```bash
LLM_CUDA_ARCH=compute_120 cargo run --release -- \
  --benchmark-load docs/serving/ps16-load.json \
  --hardware-profile docs/serving/rtx5060-ps16-hardware.cost-model.json \
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

The current runtime includes paged GQA/XQA-like decode attention, tiled
contiguous prefill attention, persistent decode buffers/execution state,
selective E4M3 decode GEMMs, and the validated scratchless atomic greedy argmax
path.

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
