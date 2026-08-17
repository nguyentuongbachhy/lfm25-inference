# MoK decode optimization benchmark gate

This document is specific to `agent/mok-fused-attention`.

The branch changes decode only. Multi-token prefill keeps the existing RMSNorm/RoPE/hybrid-attention path, so TTFT is not the primary target of this work.

## Final decode policy

The production decode path is selected from the measured `(page_size, context, batch)` region.

### Short context: one-kernel fused path

```text
raw Q / K / V
    |
    v
Q/K RMSNorm + RoPE
    |
    +--> current K/V paged-cache write
    |
    v
cp.async paged GQA
    |
    v
attention output
```

The one-kernel path avoids materializing post-RoPE Q to global memory and removes one CUDA launch. It is used only where paired ragged benchmarks showed a stable advantage.

### Medium/long context: two-kernel fast path

```text
fused Q/K RMSNorm + RoPE + KV write
    |
    v
W8 / 256-thread cp.async paged attention
    |
    v
branch-free online softmax with __expf
```

This path keeps the 256-thread staging geometry and uses CUDA fast exponential without introducing the branchy one-exp recurrence.

### Measured dispatch policy

```text
PS16:
  context <= 16                    -> one-kernel
  context <= 32 && batch <= 32     -> one-kernel
  context <= 64 && batch <= 8      -> one-kernel
  otherwise                         -> two-kernel fast

PS32:
  context <= 32 && batch <= 16     -> one-kernel
  context <= 64 && batch <= 8      -> one-kernel
  otherwise                         -> two-kernel fast
```

For ragged batches, `max_context_tokens` is derived from the host-side `positions` slice already supplied to `GpuBatch::update_step`; dispatch adds no D2H copy or synchronization.

Mixed ragged steps containing multi-token prefill segments continue to use the existing hybrid reference path.

`src/model/lfm2.rs` remains the canonical source. `build.rs` currently generates the branch-local MoK decode variant into `OUT_DIR` using guarded exact replacements; the build fails if an expected integration point is missing or duplicated. Flatten this generated integration into canonical `lfm2.rs` only after the final correctness/hardware gate.

## Promoted components

- fused Q/K RMSNorm + RoPE + KV write
- W8 / 256-thread double-buffered `cp.async` paged attention
- `__expf` fast exponential for the medium/long decode path
- one-kernel fused decode for the measured short-context region
- PS16 as the validated default deployment page size until final PS32 capacity/performance review

## Rejected experiments

These variants were removed from the source tree because paired benchmarks regressed materially:

- W4 / 128-thread async attention: staging/overlap loss at medium and long context
- branchy one-exp async attention: long-context regression despite fewer exponential calls
- global one-kernel dispatch: strong short-context win but large regression from medium/long context onward

The precise W8 async kernel remains only as a correctness/performance reference.

## Correctness gate

```bash
cargo fmt --check
cargo check --release
cargo test --release -- --test-threads=1
```

MoK-specific gates:

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

cargo test --release async_w8_fast_exp -- \
  --nocapture --test-threads=1
```

The fused single-request tests compare against the proven two-kernel MoK path across page boundaries. Ragged tests include non-contiguous physical page mappings. Fast-exp tests use non-zero K/V data and compare against the precise W8 reference.

## Retained paired benchmarks

```bash
cargo test --release bench_mok_paged_attention_paired_ab -- \
  --ignored --nocapture --test-threads=1

cargo test --release bench_mok_qk_postprocess_paired_ab -- \
  --ignored --nocapture --test-threads=1

cargo test --release bench_mok_async_w8_fast_exp_paired_ab -- \
  --ignored --nocapture --test-threads=1

cargo test --release bench_mok_one_kernel_decode_attention_paired_ab -- \
  --ignored --nocapture --test-threads=1

cargo test --release bench_mok_short_dispatch_ragged_paired_ab -- \
  --ignored --nocapture --test-threads=1
```

All paired A/B benchmarks use balanced AB/BA ordering in one process to reduce boost-clock and power-state bias.

## Final E2E gate

PS16 serving benchmark:

```bash
cargo run --release -- \
  --benchmark-serving docs/serving/mok-branch-final-ps16.json \
  --page-size 16
```

PS16 hardware benchmark:

```bash
cargo run --release -- \
  --benchmark-hardware docs/serving/mok-branch-final-hardware-ps16.json \
  --page-size 16
```

Compare against the saved `main` PS16 reports using matched points only. Prioritize:

```text
correctness NRMSE / top1
step mean / p50 / p95
output throughput / goodput
KV fragmentation and capacity
pool misses / H2D counters
prefill neutrality
```

After PS16 passes, optionally repeat hardware validation for PS32 before deciding whether PS32 should ever become the default page size.

## Merge gate

Before merging into `main` require:

1. `cargo fmt --check` and `cargo check --release` pass.
2. Full non-ignored tests pass.
3. Fused short-path, fast-exp long-path, page-boundary, and ragged correctness tests pass.
4. Model-level decode output equivalence passes across multiple decode steps.
5. Final PS16 serving/hardware E2E keeps or improves the already validated MoK gains, especially at contexts 512/2048/8192.
6. No new KV-capacity regression, pool misses, top-1 mismatch, or sequence-row numerical regression appears.
7. Prefill remains effectively neutral.
