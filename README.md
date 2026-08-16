# lfm25-inference

A Rust + CUDA inference runtime for `LFM2.5-1.2B-Instruct`, focused on low-latency GPU inference, explicit memory management, paged KV caching, and measured kernel/runtime optimization.

The project keeps BF16 as the default reference path and includes an opt-in, checkpoint- and GPU-specific selective FP8 decode path for reducing batch-1 token latency.

## Status

The runtime currently supports end-to-end `LFM2.5-1.2B-Instruct` inference, paged decode attention, contiguous prefill attention, GPU-side KV page lookup, profiling, serving, and offline FP8 calibration/evaluation workflows.

Current optimization priorities are driven by measured end-to-end latency and component-level CUDA profiling rather than isolated kernel speed alone.

## Highlights

- Rust host runtime with custom CUDA kernels and cuBLASLt GEMMs.
- BF16 reference inference path.
- Engine-level paged KV cache with configurable page size.
- XQA-like paged GQA decode with shared K/V tiles across grouped query heads.
- GPU block-table lookup for non-contiguous physical KV pages.
- Tiled contiguous prefill attention.
- Fused residual add + RMSNorm.
- Packed Gate/Up projection with a single cuBLASLt matmul.
- Packed SiLU-multiply kernel.
- Typed model weights with no hot-path string formatting or hash lookup.
- Bounded temporary-buffer pools for decode-time reuse.
- Zero-copy last-hidden-row view for the LM-head matmul.
- Coarse and detailed CUDA-event decode profiling.
- Offline checkpoint-aware FP8 calibration and quality evaluation.
- Opt-in selective E4M3 decode path with persistent quantized weights.
- In-process serving and load benchmark tooling.

## Precision Policy

BF16 is the default and golden fallback.

The current FP8 path is intentionally conservative:

- Prefill remains BF16.
- KV cache remains BF16.
- Attention math remains BF16.
- RMSNorm, RoPE, residual operations, embeddings, and sampling remain unchanged.
- Only selected `M=1` decode GEMMs use persistent tensor-wide E4M3 weights.
- FP8 is enabled only through an engine startup policy.
- Request-level precision overrides are not supported.

The promoted policy contains 16 FP8 sites:

```text
Gate/Up: layers 2, 3, 5, 7, 8, 9, 11, 15
Down:    layers 6, 8, 9, 10, 12, 14, 15
LM head
```

The policy is checkpoint- and GPU-specific. Calibration, validation, and performance benchmarking should be rerun before using it on another model checkpoint or GPU.

## Measured Performance

Measurements below were collected on an NVIDIA GeForce RTX 5060 Laptop GPU with page size 16. Laptop clocks were not locked, so FP8 results use same-process, order-balanced interleaved BF16/FP8 pairs.

### Decode TPOT

| Context | BF16 mean TPOT | FP8 mean TPOT | Paired mean speedup |
|---:|---:|---:|---:|
| 40 | 7.675 ms | 6.286 ms | 1.221x |
| 138 | 8.015 ms | 6.638 ms | 1.207x |
| 516 | 8.373 ms | 7.016 ms | 1.193x |
| 2,056 | 10.902 ms | 9.419 ms | 1.157x |
| 8,202 | 22.023 ms | 20.282 ms | 1.086x |

The reduction decreases with context length because paged attention and KV traffic account for a larger share of decode time.

### FP8 Quality Gate

The selected policy was frozen and evaluated on a disjoint WikiText-2 test split.

| Metric | Result |
|---|---:|
| Sequences | 32 |
| Next-token observations | 5,664 |
| Relative NLL delta | +0.2154% |
| Perplexity delta | +0.4335 |
| Mean KL | 0.01120 |
| Mean logit cosine | 0.998010 |
| Top-1 agreement | 92.96% |
| Final RMSNorm NRMSE | 0.08633 |
| Final RMSNorm cosine | 0.99627 |
| Non-finite logits / hidden values | 0 / 0 |

The FP8 policy is promoted only for the measured batch-1, decode-only scope. BF16 remains the default path.

## Runtime Architecture

Inference starts with tokenization and embedding lookup, then executes the LFM2.5 decoder layers before final normalization, LM-head projection, and sampling.

Each decoder layer contains three main compute paths:

- **Attention**: QKV projection, RoPE, contiguous tiled attention during prefill, paged GQA/XQA-like attention during decode, and paged KV-cache updates.
- **ShortConv**: the model-specific convolution path used by LFM2.5.
- **MLP**: packed Gate/Up projection, SiLU multiplication, and Down projection.

The decode path avoids repeated allocation and dynamic metadata lookup. Model weights are resolved into typed fields at startup, temporary buffers are reused through bounded pools, and the LM head consumes a zero-copy view of the final hidden row.

## Build and Test

Check the project:

```bash
cargo check
```

Run the correctness suite:

```bash
cargo test --release -- --test-threads=1
```

Run CUDA kernel tests:

```bash
cargo test --release cuda::kernels::tests -- --test-threads=1
```

The current regression gate is:

```text
cargo fmt --check       PASS
cargo check --release   PASS, zero warnings
cargo test --release    PASS: 54 passed, 0 failed, 11 ignored benchmarks
```

GPU correctness and benchmark tests require CUDA device access.

## Model Layout

The examples assume the model is available at:

```text
models/LFM2.5-1.2B-Instruct
```

The measured checkpoint used during the current optimization and FP8 work was:

```text
models/LFM2.5-1.2B-Instruct/model.safetensors
```

## Run Inference

Greedy end-to-end inference:

```bash
cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --prompt "Who are you?" \
  --max-new-tokens 64 \
  --page-size 16 \
  --temperature 0.0
```

Stochastic top-k sampling:

```bash
cargo run --release -- \
  --prompt "Write one sentence about CUDA." \
  --max-new-tokens 64 \
  --temperature 0.1 \
  --top-k 50 \
  --repetition-penalty 1.05 \
  --seed 42
```

`--page-size` is an engine startup policy and cannot be overridden by an individual completion request.

## Serving

Generate a scheduler hardware profile:

```bash
cargo run --release -- \
  --benchmark-hardware docs/serving/rtx5060-ps16-hardware.json \
  --page-size 16
```

Start the server:

```bash
cargo run --release -- \
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

## Profiling

Run the coarse decode profiler first:

```bash
cargo run --release -- \
  --prompt "Write the integers from 1 to 1000, separated by commas." \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --profile-decode coarse \
  --profile-warmup-steps 8 \
  --profile-steps 100
```

Use detailed profiling only after coarse profiling identifies the dominant category. Detailed mode records substantially more CUDA events and should be used for bottleneck attribution, not as the uninstrumented latency baseline.

A historical BF16 batch-1 decode profile showed the largest costs in GEMMs:

| Component | Mean per token |
|---|---:|
| MLP Gate/Up GEMM | 3.415 ms |
| MLP Down GEMM | 1.801 ms |
| LM head | 1.007 ms |
| Conv input projection | 1.018 ms |
| Attention QKV projections | 0.396 ms |
| XQA | 0.181 ms |

This is why current optimization work prioritizes precision reduction, weight reuse, batching, and multi-token execution over blindly fusing already-small kernels.

## Attention Benchmarks

Paged decode attention can be benchmarked with:

```bash
cargo test --release bench_paged_attention_lfm2_bf16 -- \
  --ignored --nocapture --test-threads=1
```

Tiled contiguous prefill attention:

```bash
cargo test --release bench_prefill_attention_lfm2_bf16 -- \
  --ignored --nocapture --test-threads=1
```

Measured batch-1 paged decode attention:

| Context | PS16 mean | PS32 mean |
|---:|---:|---:|
| 16 | 9.65 us | 13.10 us |
| 32 | 11.85 us | 13.49 us |
| 128 | 29.44 us | 37.83 us |
| 512 | 105.84 us | 104.85 us |
| 2,048 | 412.28 us | 393.83 us |

Page size should be selected on the deployment GPU with the full serving workload. Kernel-only benchmarks do not capture fragmentation or scheduler-level concurrency.

## FP8 Calibration

FP8 calibration always starts from the BF16 reference model.

Run checkpoint-aware calibration:

```bash
cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --calibrate-fp8 path/to/calibration.jsonl \
  --fp8-eval-corpus path/to/disjoint-validation.txt \
  --calibration-output docs/fp8/calibration-summary.json \
  --calibration-max-sequences 256 \
  --calibration-max-tokens 1024 \
  --fp8-eval-sequences 16 \
  --fp8-eval-max-tokens 128
```

Evaluate a frozen policy on an independent test corpus:

```bash
cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --evaluate-fp8 docs/fp8/selected-policy.json \
  --fp8-eval-corpus path/to/disjoint-test.txt \
  --fp8-eval-sequences 32 \
  --fp8-eval-max-tokens 256 \
  --evaluation-output docs/fp8/quality-final-test.json
```

Run the same-process interleaved E2E benchmark:

```bash
cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --benchmark-fp8 docs/fp8/selected-policy.json \
  --benchmark-pairs 20 \
  --benchmark-output docs/fp8/e2e-benchmark.json
```

Enable the selected decode-only FP8 policy:

```bash
cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --fp8-policy docs/fp8/selected-policy.json \
  --prompt "Who are you?" \
  --max-new-tokens 128 \
  --temperature 0.0
```

FP8 calibration and policy search are offline workflows. Production decode does not perform CPU readback or policy search.

## Benchmarking Serving

True `M=B` decode/KV/page-size benchmark:

```bash
cargo run --release -- \
  --benchmark-serving docs/serving/ps16-decode.json \
  --page-size 16
```

SLO/goodput load matrix:

```bash
cargo run --release -- \
  --benchmark-load docs/serving/ps16-load.json \
  --hardware-profile docs/serving/rtx5060-ps16-hardware.cost-model.json \
  --page-size 16
```

The in-process load benchmark excludes tokenization, HTTP serialization, and network transfer from the engine measurement.

## Development Notes

The current optimization strategy follows several rules:

1. Measure end-to-end impact before promoting a kernel or precision change.
2. Keep BF16 as a reproducible reference path.
3. Avoid allocations and dynamic metadata work in decode hot paths.
4. Reuse persistent or pooled device memory where possible.
5. Treat page size and precision as engine-level policies.
6. Validate numerical changes through downstream hidden/logit propagation, not only local kernel error.
7. Benchmark precision changes in the same process with balanced execution order when GPU clocks are not controlled.

## Documentation

Detailed records are kept under `docs/`:

```text
docs/
├── command.md
├── optimization.md
├── fp8_report.md
├── calibration.md
├── fp8/
└── serving/
```

Recommended reading:

- `docs/optimization.md`: implemented optimizations, profiling results, and current bottlenecks.
- `docs/fp8_report.md`: selective FP8 design, quality gates, and E2E performance.
- `docs/calibration.md`: calibration workload, coverage, and activation outliers.
- `docs/command.md`: correctness, benchmark, serving, profiling, and FP8 commands.

## Current Limitations

- The runtime currently targets `LFM2.5-1.2B-Instruct`.
- The promoted FP8 policy is specific to the measured checkpoint and RTX 5060 Laptop GPU.
- FP8 currently targets selected batch-1 decode GEMMs only.
- Prefill remains BF16.
- KV cache and attention math remain BF16.
- Long-context decode receives less benefit from FP8 because paged attention and KV traffic become increasingly dominant.
- The current FP8 implementation keeps BF16 weights resident and adds approximately 496 MiB of persistent FP8 weight copies for the selected 16 sites.

## Next Work

Current follow-up areas include:

- Larger-batch and continuous scheduling work.
- Lower-precision paths for larger `M`.
- Separate FP8 prefill evaluation.
- Long-context paged attention optimization.
- Reducing duplicated FP8 weight memory if memory-pressure benchmarks justify it.
- Independent evaluation of FP4 or speculative decoding.

## License

No license has been specified yet.
