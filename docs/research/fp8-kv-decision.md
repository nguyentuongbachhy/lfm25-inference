# FP8 KV decision — REJECTED

## Final decision

Reject E4M3 KV cache compression for this runtime on the RTX 5060 Laptop GPU.

Both bounded implementations passed the primitive numerical gate but failed the
performance gate. The second materially different implementation also remained
below the 1.10x context-8192 stop threshold, so the direction is closed. Do not
make a third local-kernel attempt.

## Attempt A — scalar inner-loop dequantization

| Context | NRMSE | Cosine | BF16 mean | FP8 mean | Mean speedup |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.024080 | 0.99971028 | 26.175 us | 28.323 us | 0.9336x |
| 512 | 0.023755 | 0.99971780 | 55.758 us | 62.114 us | 0.8977x |
| 2048 | 0.028125 | 0.99960466 | 211.274 us | 238.088 us | 0.8874x |
| 8192 | 0.024298 | 0.99970500 | 919.744 us | 1037.875 us | 0.8864x |

Attempt A transferred 50.2% of the BF16 KV payload but repeated scalar E4M3
conversion inside each Q-head warp.

## Attempt B — cooperative FP8x2 page decode

Attempt B moved dequantization out of the attention token loop. Each compact FP8
page was decoded cooperatively once with packed FP8x2-to-half2 conversion and
then reused from shared memory by the four Q-head warps.

| Context | NRMSE | Cosine | BF16 mean | FP8 mean | Mean speedup |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.024080 | 0.99971028 | 19.233 us | 20.717 us | 0.9369x |
| 512 | 0.023755 | 0.99971780 | 55.808 us | 67.054 us | 0.8362x |
| 2048 | 0.028125 | 0.99960466 | 211.473 us | 250.682 us | 0.8436x |
| 8192 | 0.024298 | 0.99970500 | 937.344 us | 1100.425 us | 0.8517x |

The FP8 payload remained 50.2% of BF16 at every measured context.

## Numerical result

The primitive numerical gate passed for both implementations:

- no non-finite output;
- NRMSE stayed below 0.05;
- cosine stayed above 0.999.

The identical numerical metrics between attempts confirm that Attempt B changed
only the staging/dequantization execution strategy, not the quantization format.

## Performance result

The original Phase 1 promotion targets were at least 1.15x at context 2048 and
1.25x at context 8192. The bounded stop rule required at least 1.10x at context
8192 after the final materially different implementation.

Attempt B reached only 0.8436x at context 2048 and 0.8517x at context 8192.
Therefore it fails both the promotion gate and the stop threshold.

## Root cause

Reducing global KV bytes by about half is not enough for this kernel and GPU.
The extra E4M3 decode, scale application, synchronization, and shared-memory
traffic cost more than the removed BF16 global-memory traffic. Moving packed
dequantization out of the Q-head inner loop did not reverse that tradeoff.

This result also falsifies the earlier bandwidth-only Amdahl assumption for the
current paged attention implementation: the context-sensitive decode growth is
not cheaply removable by an FP8 shadow representation on SM120.

## Continuation

Keep production BF16 KV cache and the current tuned paged/Split-K attention
path. Continue optimization from clean `main` with a different bottleneck class.
CUDA Graph launch/submission overhead is the next bounded direction.
