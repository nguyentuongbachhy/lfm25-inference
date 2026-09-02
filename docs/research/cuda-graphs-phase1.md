# CUDA Graph Phase 1 — launch/submission overhead probe

## Decision basis

The fresh decode profile measures about 7.017 ms of GPU-envelope time per decode
step. The individually measured operators account for about 6.687 ms, leaving
about 0.330 ms per step outside those regions, or roughly 4.7% of the envelope.

The current PS16 Split-K policy is already tuned from paired measurements on the
target RTX 5060 Laptop GPU, so another adaptive Split-K direction would duplicate
an optimization already present in `main`.

CUDA Graphs target a different cost class: repeated host-side submission of a
large, stable decode launch sequence.

## Hypothesis

The persistent decode executor already reuses fixed-address scratch buffers and
cached cuBLASLt plans. For a fixed decode topology, stream capture can replace
many host launch/API submissions with one graph replay and reduce launch gaps.

CUDA Graphs cannot make the model kernels themselves faster. Therefore the
expected production gain is bounded and should be largest for low-batch,
short-context decode where fixed launch overhead is the largest fraction of TPOT.

## Amdahl bound

Using only the measured 0.330 ms unaccounted portion as removable overhead gives
an ideal upper bound of approximately:

`7.017 / (7.017 - 0.330) ~= 1.049x`

This is deliberately conservative. Graph replay can also reduce CPU submission
cost that is not represented as a GPU kernel duration, but it cannot remove the
measured operator work. A production gain much above about 5% therefore needs
specific evidence rather than assumption.

## Production topology constraints

The serving runtime uses one dedicated GPU owner thread and one persistent decode
executor. This is compatible with the CUDA Graph requirement that a graph object
not be accessed concurrently from multiple threads.

A production graph cannot include `BatchModelCache::prepare_ragged`. That phase
updates host allocator state and uploads per-step token, position, slot, and block
table metadata. Production capture must keep metadata preparation outside the
graph and capture only the prepared GPU forward path.

A single universal decode graph is also invalid. Host dispatch changes the
attention launch topology as batch size and context cross MOK, unsplit, and
Split-K policy boundaries. A later production cache must therefore key graphs by
at least fixed batch size and attention execution bucket/split count.

## Phase 1 scope

Phase 1 is a compatibility and launch-overhead probe only. It does not change
production dispatch.

The ignored test `bench_cuda_graph_decode_shaped_launch_chain` uses:

- LFM2 hidden width 2048;
- persistent preallocated BF16 device buffers;
- a warm cuBLASLt plan;
- 32 repeated BF16 cuBLASLt GEMMs;
- 32 captured GPU submissions in one graph;
- thread-local stream capture;
- same-process balanced direct/graph benchmarking;
- separate host submission-time measurement;
- exact BF16 output equality after replay.

The probe focuses on cuBLASLt because GEMM submissions dominate the decode launch
sequence. Full-model Phase 2 will add the custom CUDA kernels if this primitive
gate passes.

## Compatibility attempt 0 — cudarc event-tracking isolation

The first probe failed before graph instantiation with:

`CUDA_ERROR_STREAM_CAPTURE_ISOLATION: dependency created on uncaptured work in another stream`

This was not evidence that cuBLASLt cannot be captured. `CudaRuntime` creates a
non-default stream, which makes cudarc manage `CudaSlice` access with CUDA events.
`DevicePtr` and `DevicePtrMut` then insert waits on previously recorded events.
During stream capture, those waits cross the capture boundary and CUDA correctly
rejects the dependency.

The compatibility fix does not modify production `CudaRuntime`. The Phase 1 probe
now creates a dedicated CUDA context/stream, disables cudarc event tracking before
creating the stream or any device allocation, and uses explicit single-stream
synchronization. This is safe for the isolated test topology because no second
stream can access the probe allocations.

If capture still fails after this event-tracking fix, treat the new error as the
actual CUDA/cuBLASLt compatibility result and apply at most one further fix if it
has a distinct root cause.

## Phase 1 gates

The probe must satisfy all of the following before full-model integration:

1. Stream capture and graph instantiation succeed on the target CUDA 12.8 / SM120
   environment.
2. Graph replay produces exactly the same BF16 output as the direct launch chain.
3. The paired benchmark shows at least 1.10x mean speedup for this deliberately
   launch-heavy chain, or host submission time improves by at least 3x with no GPU
   regression.

This is only a go/no-go gate. Passing it does not promote CUDA Graphs to
production.

## Phase 2 full-model gate

If Phase 1 passes, integrate capture around `DecodeExecutor::forward_prepared`
while leaving `prepare_ragged` outside capture. Cache graph instances only for
stable decode topology keys.

Before production promotion, require same-process order-balanced full-model
measurements with the selected weight-E4M3 policy:

- exact sampled-token trace agreement for deterministic decode;
- no model-quality change, because graph replay must be numerically identical;
- at least 3% mean TPOT improvement at B1/C128;
- at least 2% mean TPOT improvement at B1/C2048;
- no material regression at long context or production batch sizes;
- materially lower decode CPU submission time.

The existing NLL/hidden quality gate remains mandatory if any implementation
change affects numerical execution rather than only replay.

## Stop condition

Reject CUDA Graphs if capture is unsupported after the bounded compatibility
attempts.

If Phase 1 shows little launch/submission benefit, do not build a production graph
cache.

If Phase 1 passes but full-model B1/C128 speedup is below 1.02x in a paired test,
reject the direction because graph-cache complexity is not justified by the
measured gain.

Do not change attention math, precision, or quality thresholds to make this
direction pass.

## Commands

```bash
LLM_CUDA_ARCH=compute_120 cargo fmt --check
LLM_CUDA_ARCH=compute_120 cargo check --all-features
LLM_CUDA_ARCH=compute_120 cargo test --release -- --test-threads=1

LLM_CUDA_ARCH=compute_120 cargo test --release \
  bench_cuda_graph_decode_shaped_launch_chain -- \
  --ignored --nocapture --test-threads=1
```
