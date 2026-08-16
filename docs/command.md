# 1. Test harnesses:

```bash
# 1. compile/check all
cargo check

# 2. correctness all
cargo test --release -- --test-threads=1

# 3. correctness specific kernel
cargo test --release cuda::kernels::tests -- --test-threads=1

# 4. semantic specific op
cargo test --release residual_add_handles_vector_body_and_scalar_tail -- --test-threads=1

# 5. end-to-end LFM2.5 inference
cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --prompt "Who are you?" \
  --max-new-tokens 64 \
  --page-size 16 \
  --temperature 0.0

# 6. paged KV writer benchmark (PS16 + PS32)
cargo test --release bench_kv_cache_write_lfm2_bf16 -- \
  --ignored --nocapture --test-threads=1

# 6b. paged XQA decode benchmark (PS16 + PS32, context 16..2048)
cargo test --release bench_paged_attention_lfm2_bf16 -- \
  --ignored --nocapture --test-threads=1

# 6c. tiled contiguous prefill attention benchmark
cargo test --release bench_prefill_attention_lfm2_bf16 -- \
  --ignored --nocapture --test-threads=1

# 7. stochastic top-k sampling
cargo run --release -- \
  --prompt "Write one sentence about CUDA." \
  --max-new-tokens 64 \
  --temperature 0.1 \
  --top-k 50 \
  --repetition-penalty 1.05 \
  --seed 42

# 7b. decode breakdown profiler (disabled by default)
cargo run --release -- \
  --prompt "Write the integers from 1 to 1000, separated by commas." \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --profile-decode coarse \
  --profile-warmup-steps 8 \
  --profile-steps 100

# Use --profile-decode detailed only after coarse mode identifies the large
# category. Detailed mode adds substantially more CUDA event instrumentation.

# 8. Generate a target-GPU scheduler profile, then start the continuous server.
# Repeat with --page-size 32 before selecting the deployment policy.
cargo run --release -- \
  --benchmark-hardware docs/serving/rtx5060-ps16-hardware.json \
  --page-size 16

cargo run --release -- \
  --serve 127.0.0.1:8080 \
  --hardware-profile docs/serving/rtx5060-ps16-hardware.cost-model.json \
  --page-size 16

curl http://127.0.0.1:8080/health

curl -X POST http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Who are you?","max_new_tokens":32,"temperature":0.0}'

# 9. Offline checkpoint-aware FP8 calibration
# CORPUS accepts plain text, JSON strings, or JSONL objects with `text`/`prompt`.
cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --calibrate-fp8 path/to/calibration.jsonl \
  --fp8-eval-corpus path/to/disjoint-validation.txt \
  --calibration-output docs/fp8/calibration-summary.json \
  --calibration-max-sequences 256 \
  --calibration-max-tokens 1024 \
  --fp8-eval-sequences 16 \
  --fp8-eval-max-tokens 128

# 10. Independent validation of the frozen selected policy
cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --evaluate-fp8 docs/fp8/selected-policy.json \
  --fp8-eval-corpus path/to/disjoint-test.txt \
  --fp8-eval-sequences 32 \
  --fp8-eval-max-tokens 256 \
  --evaluation-output docs/fp8/quality-final-test.json

# 11. Same-process, order-balanced, interleaved E2E benchmark
cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --benchmark-fp8 docs/fp8/selected-policy.json \
  --benchmark-pairs 20 \
  --benchmark-output docs/fp8/e2e-benchmark.json

# 12. Opt-in production decode-only FP8; prefill remains BF16
cargo run --release -- \
  --model models/LFM2.5-1.2B-Instruct \
  --fp8-policy docs/fp8/selected-policy.json \
  --prompt "Who are you?" \
  --max-new-tokens 128 \
  --temperature 0.0

# 13. True M=B decode/KV/page-size benchmark.
cargo run --release -- \
  --benchmark-serving docs/serving/ps16-decode.json \
  --page-size 16

# 14. SLO/goodput matrix: prompt 32..8192, concurrency 1..64,
# mixed 80/20 and Poisson arrivals. This is an in-process engine benchmark;
# tokenization, HTTP serialization and network transfer are out of scope.
cargo run --release -- \
  --benchmark-load docs/serving/ps16-load.json \
  --hardware-profile docs/serving/rtx5060-ps16-hardware.cost-model.json \
  --page-size 16

# 15. M>1 tensorwide E4M3 experimental gate. Output remains NOT PROMOTED
# until the separate checkpoint-corpus quality gate and goodput gate pass.
cargo run --release -- \
  --benchmark-batched-fp8 docs/fp8/selected-policy.json \
  --benchmark-pairs 20 \
  --benchmark-output docs/fp8/batched-e4m3.json
```

`--page-size` is an engine startup policy. It is intentionally not accepted by
the per-request `/v1/completions` API.

FP8 calibration always starts from the BF16 reference model. CPU readback and
policy search exist only in explicit offline modes. Production FP8 is enabled
only by the startup `--fp8-policy` option: prefill and sensitive sites remain
BF16, selected M=1 decode GEMMs use persistent E4M3 weights, and request JSON
cannot override precision or page size.
