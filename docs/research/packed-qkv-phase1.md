# Packed QKV Phase 1 — operator-group fusion entry point

## Final status

**REJECT.**

The full-model Attempt B gate failed at the primary B1 serving shapes. See
`docs/research/packed-qkv-decision.md` for the final measurements and decision.

The direction is closed under the current research policy. Do not add a third
packed-QKV local implementation and do not integrate this branch into production.

## Baseline

This branch starts from validated `main` at
`d3c8bd8c9ccf4f8ebfa0db6f21519a37e1d67795`.

It does not include the CUDA Graph research branch. The purpose is to measure one
independent operator-group change before attempting a persistent LLM megakernel.

The LFM2 decode attention projection currently submits three separate BF16 or FP8
linear operations for Q, K, and V. For the supported model dimensions:

- hidden width: 2048;
- Q width: 2048;
- K width: 512;
- V width: 512;
- packed QKV width: 3072.

## Hypothesis

For decode, especially B1, one `[M,2048] x [3072,2048]` cuBLASLt projection can
be faster than three independent projections with output widths 2048, 512, and
512. It removes two GEMM submissions and gives the library a wider N dimension.

Attempt A uses one small BF16 unpack kernel after the packed GEMM. It copies the
3072 packed output elements per token into the existing Q/K/V scratch tensors so
all downstream QK normalization, RoPE, KV write, attention, and output projection
code remains unchanged.

This is intentionally not yet a true megakernel. If this bounded change cannot
win, a more invasive packed-QKV attention fusion is not justified.

## Expected benefit and Amdahl bound

The previous detailed decode profile measured QKV projection at approximately
306.954 us per step from a roughly 7.017 ms GPU envelope, or about 4.4%.

If the QKV region became infinitely fast, the whole-step upper bound would be
approximately:

`7.017 / (7.017 - 0.307) ~= 1.046x`.

A realistic packed-QKV change therefore targets a 1-3% full-model improvement.
The primitive projection group itself must improve much more than that to justify
extra packed weights and dispatch logic.

## Implementation A

The implementation is opt-in with:

`LFM25_PACKED_QKV=1`

When enabled, `DecodeExecutor` packs Q/K/V BF16 weights once when it is created.
It allocates one persistent `[maximum_tokens,3072]` BF16 output scratch buffer and
prepares the matching cuBLASLt plans.

For an attention layer whose Q, K, and V weights remain BF16, decode executes:

1. one packed QKV BF16 GEMM;
2. `unpack_qkv_bf16`;
3. the existing QK/RoPE/KV-write and attention path unchanged.

If Q, K, or V has an FP8 weight, the layer falls back to the existing separate
projection path. The current selected E4M3 production policy does not quantize
Q/K/V, so the packed path is compatible with that policy.

The implementation duplicates Q/K/V BF16 weights while the opt-in experiment is
enabled. This memory cost is acceptable for Phase 1 measurement but must be
removed or justified before production promotion.

## Microbenchmark

Ignored test:

`bench_packed_qkv_bf16`

Shapes:

- M1: primary decode gate;
- M8: secondary serving signal;
- M16: secondary serving signal.

Reference:

- Q GEMM `[M,2048] x [2048,2048]`;
- K GEMM `[M,2048] x [512,2048]`;
- V GEMM `[M,2048] x [512,2048]`.

Candidate:

- packed GEMM `[M,2048] x [3072,2048]`;
- one BF16 unpack kernel.

The benchmark uses same-process balanced AB/BA ordering with 20 warmup pairs, 40
measured pairs, and 20 iterations per measured batch.

## Numerical gate

For Q, K, and V outputs:

- non-finite values: 0;
- maximum NRMSE: <= 0.01;
- minimum cosine similarity: >= 0.9999.

## Primitive performance gate

Continue the direction only if M1 mean speedup is at least 1.10x.

M8 and M16 are diagnostic. A batch regression does not automatically reject a
B1-only candidate, but it prevents universal dispatch.

## Attempt A result — PASS

Measured on the target RTX 5060 Laptop GPU:

| M | Direct mean | Packed+unpack mean | Mean speedup | NRMSE | Cosine | Non-finite |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 48.074 us | 28.671 us | 1.6848x | 0.00000000 | 1.00000000 | 0 |
| 8 | 46.923 us | 21.947 us | 2.1576x | 0.00414138 | 0.99999142 | 0 |
| 16 | 50.617 us | 23.797 us | 2.1290x | 0.00414292 | 0.99999142 | 0 |

M1 exceeds the predefined 1.10x continuation gate by a large margin and is
bit-identical in this primitive test. M8 and M16 also pass the numerical gate and
show larger local speedups. The direction therefore continues to Attempt B.

## Attempt B — packed QKV consumed by QK/RoPE/KV-write

Attempt B removes the unpack launch from the normal paged-attention path.

The packed postprocess kernel consumes `[Q|K|V]` directly:

1. Q is RMS-normalized and RoPE-rotated into the existing Q scratch tensor;
2. K is RMS-normalized and RoPE-rotated directly into the paged KV cache;
3. V is copied directly from the packed projection into the paged KV cache;
4. separate K and V scratch tensors are not written on this path.

The existing short-context MoK path still uses the unpack fallback because the
MoK kernel consumes raw Q, K, and V in one fused launch. The original separate
Q/K/V projection path is also retained as a fallback for layers where any of the
Q/K/V weights use FP8.

This is a bounded operator-group fusion. Attention math, Split-K policy, KV page
layout, precision policy, and sampling are unchanged.

## Full-model trace mismatch diagnostic

The first full-model ABBA run stopped at B1/C128 because direct and packed greedy
sample traces were not bit-identical.

This does not by itself prove a packed-postprocess error. A packed 3072-wide
cuBLASLt GEMM can use a different algorithm or accumulation order than separate
2048/512/512 GEMMs. Small BF16 differences can therefore change an argmax even
when the numerical quality remains acceptable.

To isolate the source, the branch contains:

`packed_qk_postprocess_matches_unpacked_path_exactly`

This test feeds the same packed QKV tensor into two paths:

1. unpack Q/K/V, then run the existing QK/RoPE/KV-write kernel;
2. run the packed QK/RoPE/KV-write kernel directly.

It compares rotated Q, paged K cache, and paged V cache exactly.

The full-model ABBA benchmark reports `top1_match_ratio` and `first_divergence`
instead of aborting before performance measurements. Greedy trace agreement is a
diagnostic for this direction, not a replacement for the model-quality gate.

## Model-quality gate

The existing production model-quality gate remains unchanged:

- relative NLL delta <= 1%;
- no non-finite values;
- final hidden cosine >= 0.99;
- final hidden NRMSE <= 0.10.

Packed GEMM grouping is allowed to change floating-point reduction order. A
non-bit-identical greedy trace therefore requires the normal quality evaluation
before promotion; it is not automatically classified as a kernel failure.

## End-to-end gate

Ignored test:

`bench_packed_qkv_full_model_abba`

The test loads the selected E4M3 policy, uses PS16, creates fresh deterministic
prefill/cache state for every pass, and compares complete decode passes in
D/P/P/D order.

Shapes:

- B1/C128;
- B16/C128;
- B1/C2048;
- B8/C2048;
- B1/C8192.

Minimum continuation targets:

- B1/C128 mean TPOT speedup >= 1.02x;
- no p95 regression greater than 1% at B1/C128;
- B1/C2048 mean speedup >= 1.01x, or classify the candidate as short-context-only;
- no material regression at measured B8/B16 serving points if universal dispatch
  is considered;
- B1/C8192 may become neutral as attention dominates, but must not regress by
  more than 1% for a universal B1 policy.

## Attempt B full-model result — FAIL

| Shape | Mean speedup | p95 speedup | Top-1 match |
|---|---:|---:|---:|
| B1/C128 | 1.0001x | 0.9368x | 95.8333% |
| B16/C128 | 1.0288x | 1.0531x | 93.4896% |
| B1/C2048 | 0.9987x | 0.9901x | 91.6667% |
| B8/C2048 | 1.0093x | 1.0040x | 95.8333% |
| B1/C8192 | 1.0291x | 1.0945x | 100.0000% |

The B1/C128 mean gate fails and its p95 regresses materially. B1/C2048 also
fails the mean continuation gate. These performance failures are sufficient to
reject the direction; a separate model-quality run is not needed for promotion.

## Stop condition and iteration budget

Attempt A passed locally. Attempt B failed end-to-end.

The predefined stop condition now applies:

- no third local packed-QKV implementation;
- no production integration;
- no merge to main.

See `docs/research/packed-qkv-decision.md` for the final decision record.
