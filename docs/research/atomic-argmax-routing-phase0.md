# System-wide Atomic Argmax Routing — Phase 0

## Baseline

BASE_BRANCH: agent/ragged-flash-attention
BASE_SHA: 1ba0c7c6a9a7a92c3f1591f422204c32189d2d0b
RESEARCH_BRANCH: agent/atomic-argmax-routing

Previous investigations established that:
- The scratchless multi-CTA atomic argmax kernel (`argmax_rows_bf16_atomic`) runs with 32 blocks (8,192 threads) per row and achieves a 1.23x p95 speedup (saving ~254 µs per token at B=1) over the legacy single-block serial reduction (`argmax_rows_bf16`).
- However, currently `ops::argmax_bf16` and `ops::argmax_rows_bf16` in `src/ops/sampling.rs` still invoke the legacy single-block serial kernel (`launch_argmax_bf16` and `launch_argmax_rows_bf16`) directly, completely bypassing `src/ops/sampling_dispatch.rs`.
- As a result, single-token greedy generation in `src/generation/sampler.rs` and prefill fallback in `src/engine/serving/radix_owner.rs` continue to incur ~211 µs of latency per step on the single-block legacy reduction.

## Direction

DIRECTION: System-wide Atomic Argmax Dispatch & Routing
HYPOTHESIS: By routing all public `ops::argmax_bf16` and `ops::argmax_rows_bf16` invocations through `src/ops/sampling_dispatch.rs`, all single-token greedy sampling, engine fallback paths, and batched argmax callers will automatically benefit from 32-CTA parallel reduction with bitwise-identical tie-breaking and exact token agreement, shaving ~200–250 µs per token off greedy decode steps.
AFFECTED_RUNTIME_FRACTION: ~3% of decode envelope; up to 100% of sampling kernel time.
EXPECTED_LOCAL_SPEEDUP: 2.0x – 4.0x at the argmax sampling primitive layer.
AMDAHL_E2E_CEILING: 1.02x – 1.04x for end-to-end greedy decode TPOT.

## Precommitted Gates

PRIMITIVE_GATE: Paired speedup >= 1.50x for single-row (B=1) and batched (B=4, 16) argmax on vocabulary dimension 32,768–65,536
NUMERICAL_GATE: Exact bitwise match with legacy argmax output across all tested rows and distributions; 100% top-1 match
E2E_GATE: E2E greedy sampling latency reduction >= 100 µs/token
STOP_CONDITION: Any top-1 prediction mismatch or speedup < 1.00x
ITERATION_BUDGET: 2 iterations

## Benchmark Protocol

- Balanced same-process ABBA GPU timing (`benchmark_gpu_paired`).
- Single-row ($B=1, V=65,536$), batched ($B=4, V=65,536$), ($B=16, V=65,536$).
- Verify exact equality with legacy output under identical random logit distributions.
