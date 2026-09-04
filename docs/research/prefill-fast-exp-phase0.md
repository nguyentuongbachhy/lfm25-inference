# Fast-exp contiguous prefill attention — Phase 0

## Baseline

This direction starts from current `main`. Q4 prefill query reuse was rejected after exact-output benchmarks showed only ~1.045x mean speedup at both 2K and 8K despite approximately halving repeated K/V tile loads.

That result indicates repeated K/V global traffic is not the dominant long-context cost of the current contiguous prefill kernel.

The production Q2 prefill kernel performs a branch-free online softmax recurrence and currently calls precise `expf` twice per visited key.

The production medium/long decode attention path already uses the same recurrence with CUDA `__expf`, and its fast-exp numerical policy is already validated in production decode.

## Hypothesis

Long-context contiguous prefill is substantially limited by exponential throughput inside the online-softmax loop.

Replace only:

```text
expf -> __expf
```

while preserving:

- Q2 query tile;
- 32-token K/V tile;
- 256-thread CTA;
- GQA mapping;
- key traversal order;
- dot-product reduction order;
- online-softmax recurrence;
- BF16 input/output layout.

This isolates the math-function implementation from all memory/tiling changes.

## Prototype

Research-only CUDA module:

- `kernels/attention_prefill_fast_exp.cu`
- export `prefill_gqa_lfm2_bf16_fast_exp`

No production kernel registry or model dispatch is changed.

## Numerical gate

Compare complete BF16 output with the production precise-Q2 kernel at:

- N=512;
- N=2048;
- N=8192.

Use the same elementwise tolerance already accepted by the production decode fast-exp tests:

```text
abs(candidate - reference) <= 0.035 + 0.025 * abs(reference)
```

Also report:

- max absolute error;
- NRMSE;
- cosine similarity;
- non-finite count.

All outputs must satisfy the elementwise tolerance and contain no non-finite values before performance is considered.

## Performance benchmark

Use same-process balanced AB/BA GPU timing.

Reference:

```text
production Q2 + expf
```

Candidate:

```text
same Q2 + __expf
```

Test N=512, 2048, and 8192.

## Phase 0 continuation gate

Continue to model integration only if all are true:

- numerical gate passes at all three lengths;
- N=2048 mean speedup >=1.10x;
- N=8192 mean speedup >=1.10x;
- no material p95 regression at N=2048 or N=8192;
- N=512 mean does not regress by more than 3%.

The 1.10x long-context threshold is intentionally lower than the rejected Q4 1.15x 8K target because this experiment changes only one scalar math primitive and can be composed later with independent tiling work if it survives model-quality validation.

## Later model-quality gate

If the primitive passes, production integration remains opt-in until full-model validation confirms:

- existing teacher-forced relative NLL <=1%;
- no non-finite values;
- hidden cosine >=0.99;
- hidden NRMSE <=0.10;
- no unacceptable top-1/token divergence;
- material prompt-level latency improvement at approximately 2K and 8K tokens.

## Phase 0 result

RTX 5060 Laptop GPU, SM120, same-process balanced AB/BA timing:

| Tokens | Precise mean | Fast mean | Mean speedup | Precise p50 | Fast p50 | Precise p95 | Fast p95 | Cosine | NRMSE | Tolerance |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 512 | 1541.405 us | 1488.938 us | 1.0379x | 1537.152 us | 1524.704 us | 1573.920 us | 1529.120 us | 1.00000000 | 0.00001819 | pass |
| 2048 | 21474.515 us | 21323.754 us | 1.0071x | 21481.567 us | 21332.512 us | 21567.137 us | 21367.008 us | 1.00000000 | 0.00002699 | pass |
| 8192 | 356698.981 us | 343095.001 us | 1.0397x | 357328.003 us | 343552.734 us | 361042.450 us | 346186.737 us | 1.00000000 | 0.00003694 | pass |

Numerical tolerance:
- All three lengths pass the decode fast-exp tolerance (`abs(diff) <= 0.035 + 0.025 * abs(ref)`).
- 0 non-finite values. Cosine similarity is 1.00000000 across all lengths.

Performance evaluation:
- N=512: measured 1.0379x mean speedup (regression <= 3% gate passes).
- N=2048: required >= 1.10x, observed 1.0071x (FAIL).
- N=8192: required >= 1.10x, observed 1.0397x (FAIL).

Conclusion:
Replacing `expf` with `__expf` yields only ~0.7% speedup at N=2048 and ~4.0% at N=8192.
The online-softmax exponential throughput is not the dominant bottleneck of long-context contiguous prefill.

## Decision

**REJECT for production.**

Do not integrate fast-exp into contiguous prefill production dispatch.
Do not combine fast-exp with Q4 query reuse.

Both Q4 query reuse (~4.6%) and fast-exp scalar math replacement (~4.0%) fail to yield the required long-context speedup. Contiguous prefill requires a deeper algorithmic redesign (profiling instruction cost, warp serialization, shared-memory synchronization, or a tiled Tensor Core FlashAttention formulation) rather than local scalar or tiling micro-optimizations.

## Stop condition

If fast-exp fails either numerical tolerance or the 1.10x long-context mean gate, reject prefill fast-exp and do not combine it with Q4. A combined Q4+fast-exp experiment is only justified after fast-exp independently passes.
Fast-exp failed both N=2048 and N=8192 mean gates. Direction is closed.
