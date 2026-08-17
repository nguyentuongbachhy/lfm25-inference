# MoK-style fused attention benchmark gate

This document is specific to `agent/mok-fused-attention`.

The branch changes decode only. Multi-token prefill keeps the existing reference RMSNorm/RoPE/hybrid-attention path so TTFT should not be interpreted as a target of this optimization.

## Integrated decode path

Single-request decode:

```text
Q / K / V projection
        |
        v
fused Q RMSNorm + K RMSNorm + RoPE + paged KV write
        |
        v
cp.async double-buffered paged GQA
        |
        v
output projection
```

Continuous decode batches use the same fused postprocess and an async ragged paged-GQA kernel. Mixed ragged steps that contain a multi-token prefill segment keep the existing hybrid reference path.

The branch keeps `src/model/lfm2.rs` as the canonical source. `build.rs` generates the branch-local decode variant into `OUT_DIR` by replacing exactly two guarded integration points; the build fails if either reference pattern is missing or appears more than once.

## Correctness gate

Run formatting/build checks first:

```bash
cargo fmt --check
cargo check --release
```

Run the complete non-ignored test suite:

```bash
cargo test --release -- --test-threads=1
```

The MoK-specific correctness tests can also be run explicitly:

```bash
cargo test --release paged_attention_async_matches_sync_page_boundaries -- \
  --nocapture --test-threads=1

cargo test --release async_ragged_paged_attention -- \
  --nocapture --test-threads=1

cargo test --release fused_qk_postprocess -- \
  --nocapture --test-threads=1
```

The paged-attention gate includes contexts around page boundaries (`15/16/17`, `31/32/33`) and larger contexts. The ragged gate uses multiple requests with non-contiguous physical page mappings. The fused postprocess gate checks Q output, K cache and bit-exact V cache writes against the existing multi-kernel reference path.

## Order-balanced microbenchmarks

Both A/B benchmarks run reference/candidate in balanced `AB/BA` order inside one process to reduce GPU boost-clock and power-state bias.

Paged attention:

```bash
cargo test --release bench_mok_paged_attention_paired_ab -- \
  --ignored --nocapture --test-threads=1
```

This tests page sizes 16/32 and contexts:

```text
16, 32, 128, 512, 2048, 8192
```

Fused Q/K postprocess:

```bash
cargo test --release bench_mok_qk_postprocess_paired_ab -- \
  --ignored --nocapture --test-threads=1
```

For both tests, `paired_speedup_mean > 1.0` means the MoK candidate is faster. Prefer the paired mean/median over comparing two unrelated process runs.

## End-to-end continuous-decode A/B

The lightweight `--benchmark-serving` command currently covers batch sizes `1/2/4/8/16` at contexts `16` and `128`. Run the exact same command on `main` and on `agent/mok-fused-attention`.

Reference:

```bash
git switch main
cargo run --release -- \
  --benchmark-serving docs/serving/mok-main-ps16.json \
  --page-size 16
```

Candidate:

```bash
git switch agent/mok-fused-attention
cargo run --release -- \
  --benchmark-serving docs/serving/mok-branch-ps16.json \
  --page-size 16
```

Compare each matching point using:

```text
step_mean_ms
step_p50_ms
step_p95_ms
output_tokens_per_second
goodput_tokens_per_second
bf16_pool_misses_after_warmup
fp8_pool_misses_after_warmup
identical_sequence_row_nrmse_max
identical_sequence_top1_agreement
```

## Long-context continuous-decode A/B

For contexts where async KV staging matters most, use the hardware benchmark. Its decode matrix covers batch sizes through 64 and contexts `128/512/2048/8192`.

Reference:

```bash
git switch main
cargo run --release -- \
  --benchmark-hardware docs/serving/mok-main-hardware-ps16.json \
  --page-size 16
```

Candidate:

```bash
git switch agent/mok-fused-attention
cargo run --release -- \
  --benchmark-hardware docs/serving/mok-branch-hardware-ps16.json \
  --page-size 16
```

Compare `decode.points` at identical batch/context pairs, especially contexts `512`, `2048`, and `8192`. Repeat PS32 only if page size 32 is still a deployment candidate.

## Promotion gate

Do not promote the branch only because one isolated context is faster. A reasonable first gate is:

1. All correctness tests pass for PS16 and PS32.
2. Fused Q/K postprocess is consistently faster than the reference path.
3. Async attention wins at medium/long context and does not materially regress short context.
4. End-to-end decode p50/p95 and goodput improve on the intended serving workload.
5. No new BF16/FP8 pool misses or correctness regressions appear after warmup.

If async attention loses noticeably at very short context but wins at long context, keep both kernels and add a measured context crossover rather than forcing one implementation for every sequence length.
