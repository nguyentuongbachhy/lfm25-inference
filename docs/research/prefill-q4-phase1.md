# Q4 contiguous prefill attention — Phase 1

## Baseline

This direction starts from current `main` after the bounded CUDA Graph long-context promotion. Decode-local directions that were already closed remain closed.

The production contiguous causal GQA prefill kernel uses:

- 32 query heads;
- 8 KV heads;
- head dimension 64;
- GQA ratio 4:1;
- query tile = 2 tokens;
- key/value tile = 32 tokens;
- one 256-thread CTA per `(2-query-token tile, KV head)`.

With eight warps per CTA, Q2 maps exactly eight query/head tasks onto the eight warps. Each CTA then streams all causal K/V tiles needed by those two query tokens.

## Hypothesis

At long prompts, repeated K/V loading dominates enough of contiguous prefill attention that increasing query reuse inside one CTA can reduce latency.

The research Q4 kernel keeps two query states per warp. For each KV head, a CTA therefore handles four query tokens and sixteen query/head tasks while loading each shared-memory K/V tile once.

Compared with Q2:

```text
Q2: one K/V tile -> 2 query tokens
Q4: one K/V tile -> 4 query tokens
```

This halves CTA count and approximately halves repeated K/V global loads for the attention kernel. It does not change the per-query key traversal order or online-softmax arithmetic.

## Prototype

Research-only CUDA module:

- `kernels/attention_prefill_q4.cu`
- export `prefill_gqa_lfm2_bf16_q4`

No production kernel registry or model dispatch is changed.

The Q4 task mapping is deliberately chosen so each warp owns two query states with the same GQA q-head offset:

```text
warp 0: query offsets 0 and 2, q offset 0
warp 1: query offsets 0 and 2, q offset 1
...
warp 4: query offsets 1 and 3, q offset 0
...
```

Each state retains its own query values, online-softmax maximum, denominator, and two output accumulators in registers while K/V tiles are streamed through shared memory.

## Numerical gate

Before timing, compare complete BF16 outputs against the production Q2 kernel.

Requirement:

- bit-exact BF16 output at N=512, 2048, and 8192.

Because each query processes keys in the same order with the same float operations, any mismatch is treated as an implementation bug rather than an accepted numerical-policy change.

## Performance benchmark

Use same-process balanced AB/BA GPU timing for:

- N=512;
- N=2048;
- N=8192.

Reference is production Q2. Candidate is research Q4.

## Phase 1 gate

Continue to production integration only if all are true:

- exact output at all three lengths;
- N=2048 mean speedup >=1.10x;
- N=8192 mean speedup >=1.15x;
- no material p95 regression at N=2048 or N=8192;
- N=512 mean must not regress by more than 3%.

If Q4 fails the long-context gates, reject this query-reuse factor and do not change production dispatch.

If Q4 passes but N=512 regresses, a later production policy may restrict Q4 to sufficiently long prompts.

## Phase 1 result

RTX 5060 Laptop GPU, same-process balanced AB/BA timing:

| Tokens | Q2 mean | Q4 mean | Mean speedup | Q2 p95 | Q4 p95 | Exact |
|---:|---:|---:|---:|---:|---:|---|
| 512 | 1691.859 us | 1661.136 us | 1.0185x | 1697.024 us | 1669.600 us | yes |
| 2048 | 21821.562 us | 20880.138 us | 1.0453x | 23539.295 us | 22554.079 us | yes |
| 8192 | 353270.337 us | 337659.216 us | 1.0462x | 355641.846 us | 340025.482 us | yes |

The candidate is consistently faster and bit-exact. However, it misses both precommitted long-context performance gates:

- N=2048 required >=1.10x, observed 1.0453x;
- N=8192 required >=1.15x, observed 1.0462x.

The nearly flat ~4.5% gain from 2K through 8K is also diagnostic. Q4 approximately halves repeated K/V tile loads and halves CTA count, yet latency improves by only ~4.6%. Therefore repeated K/V global traffic is not the dominant cost of this kernel at long context. A blind Q8 sweep is not justified by these results.

## Decision

**REJECT for production.**

Do not integrate Q4 into production dispatch. Keep the production Q2 prefill kernel unchanged.

The experiment is still useful because it narrows the next optimization target: the dominant long-context cost is more likely inside the per-key compute/online-softmax loop than in repeated K/V tile loading.

## Stop condition

Q4 is closed. Do not sweep larger query tiles without a new measured bottleneck that specifically predicts a benefit from additional query reuse.
