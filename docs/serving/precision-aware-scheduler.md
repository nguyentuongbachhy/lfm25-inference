# Precision-Aware Scheduler Optimization

## Scope

This phase optimizes the single-GPU continuous scheduler after selective FP8 decode and measured split-K attention have already reduced raw decode GPU time. The goal is to convert that kernel-level headroom into end-to-end serving gain without regressing TTFT/TPOT correctness.

The implementation order is deliberately staged:

1. Phase A: correct production cost model.
2. Phase B: adaptive SLO-aware chunked prefill.
3. Phase C: mixed-step correction model and improved slack ordering.
4. Phase D: CPU/GPU overlap scheduling.

Only Phase A and Phase B are in scope for the current branch.

## External design review incorporated

The design review agreed with the root-cause diagnosis and Phase A -> B -> C -> D ordering, while adding four refinements that are treated as requirements:

- Deadline-aware budgeting must expose whether a single near-deadline request is throttling the whole batch. A strict minimum deadline is safe, but may become pathologically conservative at high concurrency. Do not replace it with a percentile until measurements justify that change.
- Do not assume a full multidimensional mixed-cost LUT is necessary. Phase C should first fit a small correction term on top of decode + prefill cost and only escalate to a full surface if residual error remains large.
- Full-model CUDA Graph capture was previously rejected by benchmark, but decode-only fixed-shape or piecewise capture remains a deferred research direction rather than a permanent rejection.
- Prefix reuse is a first-class Phase A cost input. Scheduling and admission must use effective remaining prompt work, not raw prompt length.

## Root cause

The previous scheduler profile was built from a legacy decode path and duplicated one measured curve into both `decode_bf16` and `decode_fp8`. Production serving now uses a persistent decode executor with selective FP8, fused FP8 down-projection preparation, and measured split-K attention dispatch. The old scheduler therefore overestimated actual FP8 decode latency and left prefill budget unused.

The previous prefill profile was also too sparse for the scheduler's chunk choices. In particular, a 256-token chunk could be charged close to the 512-token point even though the measured 256-token p95 is much lower.

## Production decode measurements

The production decode executor was measured using BF16/FP8 ABBA ordering on PS16. All valid measured points preserved top-1 agreement.

Representative p95 values:

| Batch / Context | BF16 p95 ms | FP8 p95 ms |
| --- | ---: | ---: |
| 1 / 128 | 7.728 | 6.032 |
| 16 / 512 | 8.587 | 6.596 |
| 16 / 2048 | 10.206 | 8.610 |
| 32 / 2048 | 14.125 | 12.320 |
| 64 / 2048 | 18.460 | 17.199 |

The valid production grid contains B={1,2,4,8,16,32,64} for contexts 128, 512 and 2048. Context 8192 was measured through B16; B32/B64 did not fit VRAM. The interactive serving profile remains limited to 2048 prompt tokens, so missing high-batch 8192 points are not used for the current serving lane.

## Dense prefill measurements

Measured single-segment BF16 prefill wall p95 on PS16:

| Tokens | Raw p95 ms | Scheduler envelope ms |
| ---: | ---: | ---: |
| 16 | 7.720 | 7.720 |
| 32 | 8.383 | 8.383 |
| 64 | 8.308 | 8.383 |
| 96 | 12.326 | 12.326 |
| 128 | 12.872 | 12.872 |
| 160 | 15.971 | 15.971 |
| 192 | 18.871 | 18.871 |
| 224 | 21.172 | 21.172 |
| 256 | 20.903 | 21.172 |
| 320 | 31.987 | 31.987 |
| 384 | 37.652 | 37.652 |
| 448 | 43.046 | 43.046 |
| 512 | 46.212 | 46.212 |

The scheduler cost model uses the monotone upper envelope instead of trusting small non-monotonic measurement dips as real speedups. This keeps admission conservative while preserving the large gap between 256-token and 512-token prefills.

## Phase A cost model

The measured candidate profile is:

`docs/serving/precision-aware-scheduler-ps16.cost-model.json`

It stores separate production `decode_bf16` and `decode_fp8` curves and the dense monotone prefill curve.

Decode interpolation preserves exact measured points and removes artificial batch/context cliffs. Prefill interpolation preserves the dense measured envelope and removes the old 128 -> 512 cost jump.

Prefix-aware effective work is already represented by `request.prefilled`: radix attach and refresh advance this field, so scheduler/admission calculations using `prompt_len - prefilled` naturally exclude reused prefix tokens.

## Phase B adaptive chunk solver

`maximum_prefill_tokens=512` is a hard ceiling, not a fixed scheduling choice.

For each step:

1. Schedule decode work first.
2. Predict decode p95 using the active precision curve.
3. Determine the effective step latency budget.
4. Compute remaining prefill budget.
5. Select the largest page-aligned aggregate prefill count whose measured/interpolated p95 fits that budget.
6. Pack queued prefills into that aggregate allowance, allowing only the final prompt tail to be shorter than a page.

The scheduler exposes deadline-limiter information in `BatchPlan` so high-load experiments can distinguish broad SLO pressure from one outlier request throttling the batch.

## Deadline caution

The current request deadline is anchored when `push_token` is called with the scheduler step timestamp. This is conservative relative to anchoring at token-ready time. Because Phase B now uses deadline slack to constrain work, high-concurrency benchmarks must explicitly inspect whether deadline limiting erases otherwise safe prefill headroom.

Do not switch from minimum deadline slack to p10 or another percentile without measurement. If a single-request limiter dominates under load, first determine whether the issue is deadline anchoring/admission timing or a genuine SLO-protection event.

## Phase C direction

Do not immediately benchmark a full `T(B_decode, C_decode, P, N_prefill)` surface. First measure a compact set of mixed batches and model:

`T_mixed = T_decode + T_prefill + correction`

If the correction term captures the interaction with sufficiently low residual error, keep it. Only build a larger LUT if the compact model is not accurate enough.

## Deferred CUDA Graph direction

Previous full-model CUDA Graph capture regressed serving performance and remains disabled. This does not exclude a future decode-only fixed-shape or piecewise graph experiment where dynamic attention/prefill work remains eager.

## Acceptance criteria

The scheduler phase is not accepted on owner wall time alone.

- Generation correctness must remain unchanged under existing gates.
- Poisson goodput must improve.
- Poisson TPOT p95 regression must remain below 2%.
- Mixed C16 goodput must improve or remain neutral.
- Mixed C64 must have no meaningful regression.
- Homogeneous P512/P1024/P2048 scenarios must not regress by more than 2%.
- Interactive TTFT p95 must not regress by more than 2%.
- BF16/FP8 pool misses after warmup must remain zero.
- KV peak allocation must not increase unexpectedly.
- Scheduler CPU overhead should remain neutral or improve.
- Normalized owner generated-token throughput should improve.
- Deadline-limited behavior must be audited separately, especially the fraction of steps limited by a single request.

## Current verdict

Phase A+B has the strongest evidence and best risk/leverage ratio. The main remaining uncertainty is not whether the old cost model is wrong; that is already measured. The remaining uncertainty is how much of the newly exposed prefill headroom can be safely consumed once deadline pressure and mixed decode/prefill interaction are present in a real workload.
