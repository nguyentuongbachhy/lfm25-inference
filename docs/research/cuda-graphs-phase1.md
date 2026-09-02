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
sequence. Full-model Phase 2 adds the custom CUDA kernels after this primitive
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
creates a dedicated CUDA context/stream, disables cudarc event tracking before
creating the stream or any device allocation, and uses explicit single-stream
synchronization. This is safe for the isolated test topology because no second
stream can access the probe allocations.

## Phase 1 result — PASS

Measured on the target RTX 5060 Laptop GPU:

- direct GPU mean: 622.682 us;
- graph GPU mean: 499.727 us;
- mean GPU speedup: 1.2464x;
- direct p50: 597.424 us;
- graph p50: 480.536 us;
- direct p95: 769.912 us;
- graph p95: 529.938 us;
- direct host submission: 376.253 us;
- graph host submission: 11.217 us;
- host submission speedup: 33.5440x;
- exact BF16 output equality: true.

The result passes both Phase 1 performance signals by a large margin. The graph
path improves mean GPU time by about 19.7% for this deliberately launch-heavy
chain and almost removes host submission cost. This does not imply a 19.7%
full-model TPOT gain because the real decode step contains much more kernel work.
It is sufficient evidence to proceed to the full-model gate.

## Phase 2 full-model gate

Phase 2 captures `DecodeExecutor::forward_prepared` while leaving
`BatchModelCache::prepare_ragged` outside capture. The first benchmark uses the
selected weight-E4M3 checkpoint path and fixed topology buckets at B1/C128 and
B1/C2048.

Each direct/graph pass starts from a fresh deterministic prefill. The graph pass
prepares logical decode step 0, captures the stable forward topology, then launches
the captured graph once to execute that step and advance KV plus recurrent
convolution state. Later steps update persistent metadata outside the graph and
replay the captured forward topology. The benchmark uses ABBA order across
complete passes and requires identical sampled-token traces.

Paged ragged attention reads the current position from device `position_ids`; the
context length is not frozen as a host launch scalar. The graph therefore remains
valid while the context grows inside one MOK/unsplit/Split-K topology bucket. The
Split-K count itself is fixed in the captured graph and must not cross a dispatch
boundary.

### Phase 2 compatibility attempt 0 — capture did not execute step 0

The first full-model run failed with:

`CUDA Graph sampled-token trace mismatch at B=1 C=128`

This was a benchmark-state bug, not a numerical CUDA Graph result. The harness
captured `forward_prepared` for logical step 0 and immediately read `sampled`, then
continued to logical step 1. Stream capture records the work into a graph; it does
not establish that captured work as executed recurrent history. The graph pass
therefore entered step 1 with KV and convolution state still at the prefill
boundary, while the direct pass had already executed step 0.

The fix explicitly launches the freshly captured graph once after instantiation
and upload. This executes logical step 0 before its sampled token is recorded and
before step 1 metadata is prepared. Direct and graph passes now advance from the
same prefill through the same logical decode history.

Before production promotion, require:

- exact sampled-token trace agreement for deterministic decode;
- no model-quality change, because graph replay must be numerically identical;
- at least 3% mean TPOT improvement at B1/C128;
- at least 2% mean TPOT improvement at B1/C2048;
- no material regression at long context or production batch sizes;
- materially lower decode CPU submission time.

The existing NLL/hidden quality gate remains mandatory if any implementation
change affects numerical execution rather than only replay.

## Stop condition

If full-model B1/C128 speedup is below 1.02x in the paired test, reject the
direction because graph-cache complexity is not justified by the measured gain.

If B1/C128 passes but B1/C2048 regresses materially, do not use a universal graph
policy. Continue only if a bounded short-context dispatch can preserve the gain.

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

LLM_CUDA_ARCH=compute_120 cargo test --release \
  bench_cuda_graph_full_model_abba -- \
  --ignored --nocapture --test-threads=1
```
