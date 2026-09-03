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

## Later E2E gate

Only after the kernel gate passes, integrate Q4 behind a controlled prefill dispatch and run whole-model prompt benchmarks using the selected E4M3 policy.

Primary prompt lengths:

- approximately 516 tokens;
- approximately 2056 tokens;
- approximately 8202 tokens.

Promotion requires a material whole-prompt improvement at 2K and 8K without quality change or a meaningful short-prompt regression.

## Stop condition

One Q4 implementation is the initial budget. If it fails because of a specific measurable register-occupancy bottleneck, one materially different query-state layout may be attempted. Otherwise reject Q4 and do not blindly sweep query-tile sizes.
