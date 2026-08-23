# Command Reference

## 1. Build and correctness

```bash
# Compile/check all
LLM_CUDA_ARCH=compute_120 cargo check --all-features

# Correctness suite
LLM_CUDA_ARCH=compute_120 cargo test --release -- --test-threads=1

# CUDA kernel tests
LLM_CUDA_ARCH=compute_120 cargo test --release cuda::kernels::tests -- --test-threads=1

# Example semantic op test
LLM_CUDA_ARCH=compute_120 cargo test --release \
  residual_add_handles_vector_body_and_scalar_tail -- \
  --test-threads=1
```

## 2. End-to-end inference

BF16 reference:

```bash
LLM_CUDA_ARCH=compute_120 cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --prompt "Who are you?" \
  --max-new-tokens 64 \
  --page-size 16 \
  --temperature 0.0
```

Selective E4M3 decode:

```bash
LLM_CUDA_ARCH=compute_120 cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --fp8-policy docs/benchmarks/fp8/selected-policy.json \
  --prompt "Who are you?" \
  --max-new-tokens 128 \
  --page-size 16 \
  --temperature 0.0
```

Stochastic top-k sampling:

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

## 3. Kernel and attention benchmarks

Paged KV writer:

```bash
LLM_CUDA_ARCH=compute_120 cargo test --release \
  bench_kv_cache_write_lfm2_bf16 -- \
  --ignored --nocapture --test-threads=1
```

Paged XQA decode:

```bash
LLM_CUDA_ARCH=compute_120 cargo test --release \
  bench_paged_attention_lfm2_bf16 -- \
  --ignored --nocapture --test-threads=1
```

Tiled contiguous prefill attention:

```bash
LLM_CUDA_ARCH=compute_120 cargo test --release \
  bench_prefill_attention_lfm2_bf16 -- \
  --ignored --nocapture --test-threads=1
```

## 4. Decode profiling

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

Use `--profile-decode detailed` only after coarse mode identifies a dominant
category. Detailed mode adds substantially more CUDA event instrumentation and
should not be used as the uninstrumented latency baseline.

## 5. Hardware profile and serving

The repository already contains the validated RTX 5060 Laptop / PS16
selective-E4M3 scheduler profile:

```text
docs/serving/fp8-splitk-hardware-ps16.cost-model.json
```

Start the matching production server:

```bash
LLM_CUDA_ARCH=compute_120 cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --serve 127.0.0.1:8080 \
  --fp8-policy docs/benchmarks/fp8/selected-policy.json \
  --hardware-profile docs/serving/fp8-splitk-hardware-ps16.cost-model.json \
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

To regenerate the profile on the active GPU using the same selective E4M3
policy:

```bash
LLM_CUDA_ARCH=compute_120 cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --fp8-policy docs/benchmarks/fp8/selected-policy.json \
  --benchmark-hardware docs/serving/rtx5060-ps16-hardware.json \
  --page-size 16
```

`--benchmark-hardware OUTPUT.json` writes both:

```text
OUTPUT.json
OUTPUT.cost-model.json
```

For the command above, the generated files are:

```text
docs/serving/rtx5060-ps16-hardware.json
docs/serving/rtx5060-ps16-hardware.cost-model.json
```

The scheduler profile must match the active precision/page-size configuration.
For a BF16-only server, regenerate the hardware profile without `--fp8-policy`
and use that BF16-generated cost model.

## 6. FP8 calibration

Calibration always starts from the BF16 reference model. The corpus accepts
plain text, JSON strings, or JSONL objects with `text`/`prompt`.

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

## 7. Independent FP8 validation

```bash
LLM_CUDA_ARCH=compute_120 cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --evaluate-fp8 docs/benchmarks/fp8/selected-policy.json \
  --fp8-eval-corpus path/to/disjoint-test.txt \
  --fp8-eval-sequences 32 \
  --fp8-eval-max-tokens 256 \
  --evaluation-output docs/benchmarks/final/quality.json
```

## 8. Interleaved FP8 E2E benchmark

```bash
LLM_CUDA_ARCH=compute_120 cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --benchmark-fp8 docs/benchmarks/fp8/selected-policy.json \
  --benchmark-pairs 20 \
  --benchmark-output docs/benchmarks/fp8/e2e-benchmark.json
```

## 9. Continuous decode benchmark

```bash
LLM_CUDA_ARCH=compute_120 cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --fp8-policy docs/benchmarks/fp8/selected-policy.json \
  --benchmark-serving docs/serving/ps16-decode.json \
  --page-size 16
```

## 10. Serving load/goodput benchmark

This is an in-process engine benchmark. Tokenization, HTTP serialization, and
network transfer are outside the measured engine path.

```bash
LLM_CUDA_ARCH=compute_120 cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --fp8-policy docs/benchmarks/fp8/selected-policy.json \
  --benchmark-load docs/serving/ps16-load.json \
  --hardware-profile docs/serving/fp8-splitk-hardware-ps16.cost-model.json \
  --page-size 16
```

## 11. Batched E4M3 research gate

The M>1 tensor-wide E4M3 path remains experimental unless a separate quality
and goodput gate promotes it.

```bash
LLM_CUDA_ARCH=compute_120 cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --benchmark-batched-fp8 docs/benchmarks/fp8/selected-policy.json \
  --benchmark-pairs 20 \
  --benchmark-output docs/benchmarks/fp8/batched-e4m3.json
```

## Notes

- `--page-size` is an engine startup policy and cannot be overridden per request.
- Production FP8 is enabled only with startup `--fp8-policy`.
- Calibration and policy search are offline workflows.
- CPU readback used by offline analysis is not part of production decode.
- GPU benchmarks should run directly on the target host with CUDA device access.
