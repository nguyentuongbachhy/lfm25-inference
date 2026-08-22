# Validated Runtime v2

## Scope

`release/validated-runtime-v2` is a clean production checkpoint built directly from `main` commit `be016b276055e3923f0ae006d1269d1c8355e67d`.

The base already contains validated runtime v1 (`9e4e51f3cf5d6d6d476c72bfba8107d7640a8460`) plus the follow-up clippy cleanup.

Runtime v2 adds only the validated scratchless atomic greedy argmax path. It intentionally excludes all rejected or unfinished model-roofline experiments.

## Runtime v1 baseline

The v1 baseline contains:

- persistent selective FP8 decode weights and activation scratch;
- fused SwiGLU-to-E4M3 for selected FP8 down projections;
- BF16 prefill preservation;
- measured PS16 Split-K attention dispatch and exact log-sum-exp merge;
- persistent bounded Split-K workspace;
- serving/runtime plumbing validated before the v1 release.

## Runtime v2 delta

The v2 delta contains:

- scratchless multi-CTA atomic argmax;
- exact legacy BF16 tie behavior encoded into the atomic key;
- exact handling of NaN, negative infinity and signed zero relative to the legacy kernel;
- no persistent argmax workspace;
- production dispatch only for `rows <= 16` and `columns <= 65536`;
- legacy fallback for larger batches or vocabularies;
- atomic path enabled by default;
- `LFM25_ATOMIC_ARGMAX=0|false|off|no` disables the production atomic path;
- unit correctness gates against the legacy implementation;
- a full-model ABBA test that compares the complete warmup plus measured sampled trace.

## Measured full-model evidence

Target device: RTX 5060 Laptop GPU, CUDA architecture `compute_120`.

The original full-model ABBA measurements used the selected FP8 policy and PS16 runtime baseline. Every reported case below had `top1_agreement=true`.

| Batch | Context | Legacy mean ms | Atomic mean ms | Mean speedup | Saving us | Legacy p95 ms | Atomic p95 ms | p95 speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 5.994935 | 5.920192 | 1.0126x | 74.743 | 6.042944 | 5.976320 | 1.0111x |
| 2 | 128 | 6.024318 | 5.943110 | 1.0137x | 81.208 | 6.073536 | 5.964448 | 1.0183x |
| 8 | 128 | 6.066702 | 5.993897 | 1.0121x | 72.805 | 6.112352 | 6.046048 | 1.0110x |
| 16 | 128 | 6.300557 | 6.239653 | 1.0098x | 60.904 | 6.525728 | 6.442944 | 1.0128x |
| 32 | 128 | 7.194995 | 7.208317 | 0.9982x | -13.322 | 7.291808 | 7.444544 | 0.9795x |
| 64 | 128 | 8.181772 | 9.366310 | 0.8735x | -1184.538 | 8.243936 | 9.429280 | 0.8743x |
| 1 | 512 | 6.353706 | 6.277812 | 1.0121x | 75.894 | 6.404576 | 6.344832 | 1.0094x |
| 8 | 512 | 6.707775 | 6.608489 | 1.0150x | 99.287 | 6.797920 | 6.677088 | 1.0181x |
| 32 | 512 | 9.091413 | 9.017204 | 1.0082x | 74.209 | 9.135776 | 9.071040 | 1.0071x |
| 1 | 2048 | 6.483673 | 6.426691 | 1.0089x | 56.981 | 6.552512 | 6.502880 | 1.0076x |
| 8 | 2048 | 8.734368 | 8.648835 | 1.0099x | 85.533 | 8.812096 | 8.735232 | 1.0088x |

## Production dispatch decision

The accepted policy is deliberately narrower than the functionally supported atomic domain:

```text
B <= 16 and vocab <= 65536 -> atomic argmax
otherwise                  -> legacy argmax
```

Reasoning:

- B1-B16 showed consistent positive mean and p95 full-model results.
- B32 was context-dependent and therefore not a stable production win.
- B64 regressed materially and is excluded.
- Correctness alone is not sufficient to extend the production performance domain.

## Correctness contract

The optimized path must match the existing runtime's legacy behavior, including its non-standard tie ordering:

1. Legacy logical lane `L` owns columns `L, L+256, L+512, ...`.
2. Equal maxima inside one lane prefer the earliest column in that lane.
3. Equal maxima across lanes prefer the smaller logical-lane id, not necessarily the numerically smaller token id.
4. NaN and `-inf` never beat the legacy `-FLT_MAX` initialization.
5. An all-ignored row falls through to token index zero.
6. `+0` and `-0` compare equal and are resolved by legacy tie priority.

The atomic key stores:

- 16 bits: monotonic BF16 value ordering;
- 8 bits: inverse logical-lane priority;
- 8 bits: inverse within-lane offset priority.

This packing is valid only through 65536 columns, which is enforced by the launcher and production dispatcher.

## Explicitly excluded research

The following experiments are not part of runtime v2:

- multi-block scratch argmax candidate;
- W8A8 tiny-M down projection;
- W8A16 tiny-M down projection;
- CTA128 attention research;
- tiny-BF16 SIMT GEMM research;
- MXFP8/block-32 research;
- precision-aware scheduler research.

Relevant conclusions from rejected experiments:

- W8A8 failed full-model decision stability despite strong local GEMM speedups.
- W8A16 layer 13 achieved approximately 2.9% B1/C128 full-decode speedup in a later ABBA run but failed sampled-token agreement, so it is rejected for production.
- tiny-BF16 SIMT GEMMs remained slower than cuBLASLt; rotating-weight measurements showed cuBLASLt at roughly 323-343 GB/s for down projection and 334-338 GB/s for gate-up, close enough to the device bandwidth roofline that a same-byte BF16 replacement has low expected ROI.
- MXFP8 is still research-only. A small block-32 capability test produced `rel_l2=0.00586357`, `cosine=0.99999324`, `max_abs=0.5`, but no production decision has been made.

## Final validation before merge

Run on the target CUDA machine from `release/validated-runtime-v2`:

```bash
LLM_CUDA_ARCH=compute_120 cargo check --all-features

LLM_CUDA_ARCH=compute_120 cargo test --release \
  atomic_argmax \
  -- --test-threads=1 --nocapture

LLM_CUDA_ARCH=compute_120 cargo test --release \
  production_policy_ \
  -- --test-threads=1 --nocapture

LLM_CUDA_ARCH=compute_120 cargo test --release \
  bench_production_atomic_argmax_abba \
  -- --ignored --test-threads=1 --nocapture
```

Merge criteria:

- compilation succeeds;
- all atomic correctness tests pass;
- dispatcher boundary tests pass;
- every full-model ABBA case reports `top1_agreement=true` for the complete sampled trace;
- no systematic mean or p95 regression appears in the B<=16 production domain.

Only after these gates should `release/validated-runtime-v2` be merged into `main`.
