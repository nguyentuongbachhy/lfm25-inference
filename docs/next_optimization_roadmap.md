# Next Optimization Roadmap

## Purpose

This document is the continuation plan after the current validated runtime milestone.
It is intentionally separate from `docs/optimization.md`, which records work that has
already been measured and integrated.

The current release should be treated as a frozen baseline. Future optimization work
must start from a fresh profile of `main`, use one research branch per direction, and
only return to production after both quality and end-to-end performance gates pass.

## Current validated baseline

Target used for the current measurements:

- Model: `LFM2.5-1.2B-Instruct`
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- Architecture: Blackwell GeForce SM120 / `compute_120`
- CUDA: 12.8.x
- BF16 prefill and golden fallback
- Selective tensor-wide E4M3 for validated M=1 decode GEMMs
- BF16 paged KV cache
- PS16 paged GQA/XQA-like decode attention
- Tiled contiguous prefill attention
- Persistent decode executor and bounded temporary pools
- Scratchless atomic greedy argmax for the validated production domain

The final independent FP8 validation is recorded in `docs/final_release_report.md`.

## Optimization rules

Every future direction should define these before implementation:

1. **Hypothesis** — what bottleneck is being reduced.
2. **Expected benefit** — the maximum plausible end-to-end gain, preferably using
   Amdahl's law or measured component time.
3. **Microbenchmark** — the smallest experiment that can falsify the hypothesis.
4. **Numerical gate** — the required local and downstream correctness/quality bounds.
5. **E2E gate** — the production workload that must improve.
6. **Stop condition** — when to reject the direction instead of continuing to tune it.
7. **Iteration budget** — avoid repeatedly changing implementation details without new
   evidence about the root cause.

Use same-process, order-balanced measurements when clocks are not controlled. Kernel
speed alone is not sufficient for promotion.

## Branch policy

Recommended branch names:

```text
agent/fp8-kv
agent/adaptive-splitk
agent/cuda-graphs
agent/continuous-batching
agent/prefix-kv
agent/fusions-<target>
agent/speculative-decode
```

Each branch should start from the latest validated `main`, not from another research
branch unless there is an explicit dependency.

Promoted work should merge production code, tests, benchmark evidence, and a short
report. Rejected work should not leave dead runtime code in `main`; keep only concise
research evidence when useful.

---

## Phase 1 — Re-profile the frozen baseline

Do this before selecting any optimization below.

### Hypothesis

The dominant bottleneck may have changed after selective FP8, Split-K attention,
persistent decode execution, and atomic argmax were integrated.

### Measure

At minimum profile:

```text
contexts: 128, 512, 2048, 8192
batch:    1, 2, 4, 8, 16 where supported
prefill and decode separately
```

Record:

- TTFT
- TPOT
- tokens/s
- MLP Gate/Up
- MLP Down
- Conv
- Attention / XQA
- LM head
- Sampling
- launch/runtime overhead
- KV-cache traffic and memory use
- temporary-pool misses

### Stop condition

Do not start a new kernel/precision project until a measured bottleneck and its
maximum plausible E2E contribution are known.

---

## Phase 2 — FP8 KV cache

### Why it may matter

The current K/V cache is BF16. As context grows, paged attention and KV traffic take a
larger fraction of decode time, which is also why the existing FP8 weight speedup
shrinks at long context.

### Hypothesis

Reducing K/V storage and bandwidth to FP8 can improve long-context decode latency and
increase usable context/batch capacity without materially changing model quality.

### Candidate designs

- E4M3 K/V with per-head scaling
- E4M3 or E5M2 with per-page scaling
- separate K and V scale policies if their distributions differ
- dequantization inside the attention load path rather than materializing BF16 K/V

### Benchmarks

Contexts:

```text
128, 512, 2048, 8192
optional: 16384, 32768 if memory permits
```

Measure:

- attention kernel latency
- full TPOT
- bytes/token
- peak KV VRAM
- maximum context/batch capacity
- quantize/dequantize overhead

### Numerical and quality gate

Compare against the BF16 KV reference using:

- attention-output NRMSE/cosine
- final hidden NRMSE/cosine
- teacher-forced NLL/PPL
- mean/p50/p95/p99 KL
- logit cosine
- top-1/top-5/top-10 agreement/overlap
- non-finite checks
- long greedy traces as diagnostics

Do not relax the current production quality envelope just to obtain a speedup.

### Stop condition

Reject if dequantization overhead erases the bandwidth saving, or if long-context
quality fails while a materially safer scaling design has already been tried.

---

## Phase 3 — Adaptive Split-K attention

### Why it may matter

The best Split-K configuration can depend on context length, batch size, page size,
head count, and target GPU. A single static policy may be suboptimal across serving
workloads.

### Hypothesis

A small startup-derived lookup/cost model can choose a better Split-K policy without
adding expensive tuning to the request hot path.

### Benchmark matrix

Vary:

```text
context
batch
page size
KV heads
candidate split counts
```

Measure kernel latency and full TPOT using interleaved comparisons.

### Production constraint

The selection logic must be O(1) or a compact table lookup during inference. Do not
run dynamic benchmark searches in the request path.

### Stop condition

Reject if the best adaptive policy produces only negligible E2E improvement or if the
static production policy already sits within measurement noise of the per-shape best.

---

## Phase 4 — CUDA Graphs, only if launch-bound

### Entry condition

Do not begin this phase unless profiling shows meaningful CPU launch/enqueue or GPU
launch-gap overhead after the existing persistent executor work.

### Hypothesis

A stable M=1 decode graph can reduce repeated launch overhead when buffer addresses and
execution structure remain fixed.

### Required design work

- stable device-buffer addresses
- graph-compatible KV metadata updates
- page/block-table handling
- precision-policy stability
- graph recapture strategy for shape changes

### Benchmark

Compare normal vs graph decode in the same process across several contexts and batch
sizes. Report TPOT and CPU submission time separately.

### Stop condition

Reject immediately if kernel execution dominates and the launch-overhead ceiling is too
small to justify graph-management complexity.

---

## Phase 5 — Continuous batching

### Why it may matter

Single-request TPOT is not the same problem as production throughput. Weight reuse
across concurrent requests can improve GPU utilization and amortize GEMM traffic.

### Required capabilities

- requests with independent sequence lengths
- paged KV allocation/reclamation
- decode scheduling
- prefill/decode coexistence
- bounded queueing latency
- cancellation/error cleanup
- deterministic ownership of KV pages and temporary buffers

### Workloads

Benchmark at least:

```text
batch/concurrency: 1, 2, 4, 8, 16, 32, 64
contexts:          128, 512, 2048, 8192
```

Include:

- closed-loop saturation
- Poisson arrivals
- mixed short/long prompts
- prefill-heavy and decode-heavy mixtures

Measure:

- throughput
- goodput under SLO
- TTFT p50/p95
- TPOT p50/p95
- queueing delay
- VRAM
- KV fragmentation

### Stop condition

Do not optimize maximum throughput at the cost of uncontrolled p95 latency. The
scheduler should be judged by goodput under an explicit latency objective.

---

## Phase 6 — Prefix KV reuse

### Target workloads

This is useful when requests share long exact prefixes, for example:

- system prompts
- agent instructions
- RAG templates
- repeated conversation prefixes

### Correctness requirements

- exact token-prefix identity
- correct positional state
- correct ShortConv/recurrent state where applicable
- ownership/reference counting
- eviction policy
- no cross-request mutation or data leakage

### Measure

- TTFT reduction
- prefill tokens avoided
- hit rate
- extra VRAM
- lookup overhead
- eviction effectiveness

### Stop condition

Reject or defer if realistic workload hit rate is too low for the memory/complexity
cost.

---

## Phase 7 — Bounded kernel fusions

Only profile-driven fusions should be attempted.

Potential candidates include:

- SwiGLU -> activation quantization
- RMSNorm -> quantization
- residual add + normalization extensions
- QKV preprocessing
- sampling pipeline cleanup

For each candidate, calculate its current absolute time first. Do not build a large
monolithic kernel around a component whose theoretical E2E contribution is negligible.

### Stop condition

If two materially different implementations fail for the same fundamental reason, or
if the component already meets the E2E target, stop rather than continuing to tune it.

---

## Phase 8 — Speculative decoding

This should remain late in the roadmap.

### Prerequisites

The runtime needs robust rollback semantics for:

- KV cache
- ShortConv/model state
- scheduler state
- sampling state
- page allocation

### Evaluate

- draft-model cost
- acceptance rate
- verification cost
- rollback cost
- additional VRAM
- actual TPOT/E2E improvement

For a 1.2B target model, speculation may not be beneficial because the target itself is
already relatively small. Require an early bounded experiment before investing in a
full scheduler integration.

### Stop condition

Reject if expected accepted tokens per verification step do not compensate for draft,
verification, and rollback overhead.

---

## Closed precision directions from the current campaign

The following should not be reopened without materially new evidence, hardware, CUDA
support, or a different quantization design:

- W8A8 tiny-M path — rejected on model-level correctness/stability
- W8A16 tiny-M path — local performance improvement but model-level decision stability
  failed
- custom tiny-M BF16 GEMM — cuBLASLt was already close to the available bandwidth
  roofline
- MXFP8 block-32 — slower than the existing tensor-wide E4M3 path on the measured GPU
- NVFP4 SM120 — strong primitive performance, but the final disjoint-test propagation
  confirmation rejected every surviving production candidate

See `docs/research/nvfp4_rejection.md` for the NVFP4 decision record.

## Suggested order

Unless a fresh profile strongly changes the ranking:

```text
re-profile main
    -> FP8 KV cache
    -> adaptive Split-K attention
    -> CUDA Graphs only if launch-bound
    -> continuous batching
    -> prefix KV reuse
    -> bounded fusions
    -> speculative decoding
```

The roadmap is not a checklist that must be completed. Each phase exists to answer a
specific performance question. If its gate fails, record the evidence and move on.
