# FP8 KV Phase 1 — PS16 E4M3 primitive

## Decision basis

The fresh production re-profile shows that selective E4M3 decode TPOT grows from
6.240 ms at 40 prompt tokens to 13.032 ms at 8202 prompt tokens. The
context-sensitive increase is 6.792 ms, or about 52.1% of the long-context TPOT.
The existing weight-only FP8 speedup falls from 1.232x to 1.054x over the same
range.

This makes KV/attention traffic the highest-value next precision target. CUDA
Graphs cannot remove context-proportional memory traffic, and the MLP path is
already the dominant short-context region covered by the selected weight FP8
policy.

## Hypothesis

BF16 K/V traffic limits long-context decode. E4M3 K/V with bounded scaling can
reduce the bytes read by paged attention enough to offset dequantization cost on
SM120.

## Expected benefit

The FP8 payload is approximately half the BF16 K/V payload. Per-page/per-KV-head
scales add less than 1% metadata for PS16.

If the entire measured 6.792 ms context-sensitive increase at 8202 tokens were
bandwidth-bound and halved, the ideal Amdahl upper bound would be about 1.35x
for the production decode TPOT. This is an upper bound, not a prediction.

## Phase 1 scope

This phase is intentionally isolated from production cache ownership:

- page size: PS16 only;
- format: E4M3 only;
- scale scope: one K scale and one V scale per physical page and KV head;
- BF16 cache remains the golden source;
- an FP8 shadow cache is built only for the primitive experiment;
- single-sequence paged GQA/XQA-like attention only;
- no ragged serving integration;
- no production dispatch change.

The shadow-cache design avoids an invalid experiment where a scale changes while
an incrementally filled page still contains bytes encoded with an older scale.
If the primitive is promoted, incremental production writes need a separate
layout/scale ownership design.

## Numerical gate

For contexts 128, 512, 2048 and 8192, compared with the existing BF16
async-fast paged attention output:

- no non-finite values;
- output NRMSE <= 0.05;
- output cosine >= 0.999.

These are primitive gates only. They do not replace the model-quality gate.

## Performance gate

Use same-process, order-balanced paired GPU measurements.

Phase 1 is promising only if:

- context 2048 mean attention speedup >= 1.15x; and
- context 8192 mean attention speedup >= 1.25x.

A short-context regression is acceptable only if a later bounded dispatch can
exclude that domain.

## Attempt A — scalar inner-loop FP8 dequantization

Attempt A staged compact FP8 K/V pages in shared memory and converted FP8 values
to float inside the attention inner loop.

Measured results on the RTX 5060 Laptop GPU:

| Context | NRMSE | Cosine | BF16 mean | FP8 mean | Mean speedup |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.024080 | 0.99971028 | 26.175 us | 28.323 us | 0.9336x |
| 512 | 0.023755 | 0.99971780 | 55.758 us | 62.114 us | 0.8977x |
| 2048 | 0.028125 | 0.99960466 | 211.274 us | 238.088 us | 0.8874x |
| 8192 | 0.024298 | 0.99970500 | 919.744 us | 1037.875 us | 0.8864x |

The primitive numerical gate passes at every measured context. The FP8 payload
is 50.2% of BF16 including scale metadata. However, performance fails at every
context and context 8192 is below the 1.10x stop threshold.

The main implementation issue is repeated dequantization. LFM2 GQA has four Q
heads per KV head. Attempt A makes each active Q-head warp convert the same K/V
FP8 data independently while traversing a page. The saved global-memory traffic
therefore does not compensate for the repeated scalar conversion work.

## Attempt B — cooperative FP8x2 page decode

The one allowed materially different implementation changes the dequantization
placement instead of tuning minor launch parameters.

Attempt B:

- still transfers compact FP8 K/V from global memory;
- keeps the same per-page/per-KV-head scales and numerical format;
- uses CUDA packed FP8x2-to-half2 conversion;
- cooperatively decodes each staged physical page once;
- stores the decoded page in shared half precision;
- reuses that decoded page across all four Q-head warps;
- keeps global FP8 staging double-buffered so the next compact page transfer can
  overlap current-page attention work.

This removes repeated E4M3 conversion from the attention token loop. FP8 values
are exactly representable in FP16 before applying the float scale, so this
change is not expected to add meaningful quantization error relative to Attempt
A. The numerical gate remains unchanged and must still be measured.

Attempt B is the final bounded dequantization/staging attempt for this direction.
If it does not reach at least 1.10x mean speedup at context 8192, reject FP8 KV
for this runtime instead of continuing local kernel tuning.

## Model-quality gate before production promotion

Production integration still requires the existing hard gate:

- relative NLL delta <= 1%;
- no non-finite values;
- final hidden cosine >= 0.99;
- final hidden NRMSE <= 0.10.

It also requires held-out teacher-forced evaluation, long behavioral diagnostics,
and same-process E2E benchmarking.

## E2E gate before production promotion

After a primitive pass, the production candidate must improve selective-E4M3
TPOT by at least:

- 5% at context 2048; and
- 10% at context 8192,

without failing the model-quality gate.

## Stop condition

Reject the E4M3 KV direction if the numerical gate fails materially.

If quality passes but context 8192 primitive speedup is below 1.10x, allow at
most one materially different dequantization/staging implementation. Reject the
direction if both implementations fail for the same dequantization-overhead
root cause.

Attempt B is that final allowed implementation.

Do not tune small parameters indefinitely.

## Commands

```bash
LLM_CUDA_ARCH=compute_120 cargo fmt --check
LLM_CUDA_ARCH=compute_120 cargo check --all-features
LLM_CUDA_ARCH=compute_120 cargo test --release -- --test-threads=1

LLM_CUDA_ARCH=compute_120 cargo test --release \
  fp8_kv_attention_quality_smoke_ps16 -- \
  --ignored --nocapture --test-threads=1

LLM_CUDA_ARCH=compute_120 cargo test --release \
  bench_fp8_kv_attention_ps16 -- \
  --ignored --nocapture --test-threads=1
```

Do not proceed to ragged serving or replace the BF16 production cache until
these Phase 1 results are recorded and evaluated.
