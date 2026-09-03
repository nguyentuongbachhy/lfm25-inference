# Packed QKV Phase 1 — operator-group fusion entry point

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

## Model-quality gate

If the primitive gate passes, the existing production model-quality gate remains
unchanged:

- relative NLL delta <= 1%;
- no non-finite values;
- final hidden cosine >= 0.99;
- final hidden NRMSE <= 0.10.

Because packed BF16 projection changes only GEMM grouping, deterministic greedy
sequence agreement should also be checked before promotion.

## End-to-end gate

If the primitive gate passes, compare selected-weight-E4M3 baseline against
selected-weight-E4M3 plus packed QKV in same-process order-balanced full-model
benchmarks.

Minimum continuation targets:

- B1/C128 mean TPOT speedup >= 1.02x;
- no p95 regression greater than 1% at B1/C128;
- B1/C2048 mean speedup >= 1.01x, or classify the candidate as short-context-only;
- no material regression at measured B8/B16 serving points if universal dispatch
  is considered.

## Stop condition and iteration budget

Attempt A is packed GEMM plus unpack.

If M1 primitive speedup is below 1.10x, reject packed QKV and do not proceed to a
persistent megakernel from this direction.

If Attempt A passes M1 but the unpack kernel is clearly the blocker in the
full-model result, allow one materially different Attempt B: make the fused
QK-normalization/RoPE/KV-write path consume the packed QKV layout directly and
remove the unpack launch plus Q/K/V copies.

No third local packed-QKV implementation is allowed before reassessing the larger
megakernel strategy.

## Commands

```bash
LLM_CUDA_ARCH=compute_120 cargo fmt --check
LLM_CUDA_ARCH=compute_120 cargo check --all-features
LLM_CUDA_ARCH=compute_120 cargo test --release -- --test-threads=1

LLM_CUDA_ARCH=compute_120 cargo test --release \
  bench_packed_qkv_bf16 -- \
  --ignored --nocapture --test-threads=1
```
