# MoK-style fused attention benchmark gate

This document is specific to `agent/mok-fused-attention`.

The branch changes decode only. Multi-token prefill keeps the existing reference RMSNorm/RoPE/hybrid-attention path so TTFT should not be interpreted as a target of this optimization.

## Integrated decode path

Single-request and decode-only ragged batches now use one fused CUDA kernel per paged-attention layer:

```text
raw Q / K / V projection outputs
        |
        v
Q RMSNorm + K RMSNorm + RoPE
        |
        +--> current K/V paged-cache write
        |
        v
cp.async double-buffered paged GQA
        |
        v
attention output
```

The one-kernel path keeps rounded Q in registers instead of materializing post-RoPE Q to global memory. The first cache page starts staging before Q/K postprocess. For a one-page context, the newly written current token is patched into the staged page after the asynchronous copy completes. Online softmax uses one `expf` per key token rather than the two-exp update used by the earlier async kernel.

The previous two-kernel MoK path remains in the branch as the correctness/performance reference:

```text
fused Q/K postprocess + KV write
        |
        v
async paged attention
```

Mixed ragged steps containing a multi-token prefill segment still use the existing hybrid reference path.

`src/model/lfm2.rs` remains the canonical source. `build.rs` generates the branch-local decode variant into `OUT_DIR` with guarded exact replacements; the build fails if an expected integration point is missing or duplicated.

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

Run MoK-specific gates explicitly:

```bash
cargo test --release paged_attention_async_matches_sync_page_boundaries -- \
  --nocapture --test-threads=1

cargo test --release async_ragged_paged_attention -- \
  --nocapture --test-threads=1

cargo test --release fused_qk_postprocess -- \
  --nocapture --test-threads=1

cargo test --release fused_decode_attention -- \
  --nocapture --test-threads=1

cargo test --release fused_ragged_decode_attention -- \
  --nocapture --test-threads=1
```

The one-kernel single-request tests compare directly against the previous two-kernel MoK path across one-page and multi-page contexts for PS16 and PS32. They compare attention output plus K/V cache contents. The ragged tests use multiple requests with non-contiguous physical page mappings and compare the same outputs/cache state.

## Order-balanced microbenchmarks

All paired A/B benchmarks use balanced `AB/BA` order inside one process to reduce laptop boost-clock and power-state bias.

Earlier async attention vs sync reference:

```bash
cargo test --release bench_mok_paged_attention_paired_ab -- \
  --ignored --nocapture --test-threads=1
```

Earlier fused Q/K postprocess vs multi-kernel reference:

```bash
cargo test --release bench_mok_qk_postprocess_paired_ab -- \
  --ignored --nocapture --test-threads=1
```

New decisive microbenchmark, previous two-kernel MoK vs one-kernel MoK:

```bash
cargo test --release bench_mok_one_kernel_decode_attention_paired_ab -- \
  --ignored --nocapture --test-threads=1
```

It covers page sizes 16/32 and contexts:

```text
16, 32, 128, 512, 2048, 8192
```

`paired_speedup_mean > 1.0` means the one-kernel candidate is faster. Prefer paired mean/p50/p95 over comparing unrelated process runs.

## End-to-end continuous-decode A/B

The lightweight `--benchmark-serving` command covers batch sizes `1/2/4/8/16` at contexts `16` and `128`.

Reference baseline should remain the saved `main` report. Re-run only if the machine/runtime state materially changed.

Candidate:

```bash
git switch agent/mok-fused-attention
cargo run --release -- \
  --benchmark-serving docs/serving/mok-branch-one-kernel-ps16.json \
  --page-size 16
```

Compare matching points using:

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

The hardware benchmark is the merge-deciding performance gate because it covers contexts `128/512/2048/8192` and batch sizes through 64 where capacity permits.

Candidate:

```bash
git switch agent/mok-fused-attention
cargo run --release -- \
  --benchmark-hardware docs/serving/mok-branch-one-kernel-hardware-ps16.json \
  --page-size 16
```

Compare it against the saved `mok-main-hardware-ps16.json`, especially contexts `512`, `2048`, and `8192`.

Run PS32 only after PS16 correctness and paired A/B pass:

```bash
cargo run --release -- \
  --benchmark-hardware docs/serving/mok-branch-one-kernel-hardware-ps32.json \
  --page-size 32
```

## Merge gate

Before merging MoK into `main` require all of the following:

1. Full non-ignored test suite passes.
2. One-kernel fused single and ragged correctness tests pass on PS16 and PS32.
3. One-kernel paired A/B is not materially slower at short context and improves medium/long contexts.
4. Hardware E2E keeps or improves the already validated two-kernel MoK gains at `512/2048/8192`.
5. No new pool misses, KV-capacity regressions, top-1 mismatches, or sequence-row numerical regressions appear.
6. Prefill remains effectively neutral because it is not a target of this optimization.

If the one-kernel path regresses short context but wins materially from a measured context threshold onward, keep the two-kernel path as a measured short-context fallback rather than reverting the fusion globally.
