# Selective FP8 decode report

## Verdict: PROMOTE

Promote `policy_frontier_16` as an opt-in, startup-selected, decode-only policy
for `LFM2.5-1.2B-Instruct` on the measured RTX 5060 Laptop GPU. It passes an
independent held-out quality gate and exceeds the required 1.20x short-context
batch-1 TPOT gate: paired mean speedup is 1.2211x for 40 prompt tokens and 128
fixed decode steps.

This verdict is deliberately scoped. BF16 remains the default golden path and
fallback. Prefill remains BF16. The FP8 policy is not claimed portable to a
different checkpoint or GPU without rerunning calibration, validation and the
interleaved benchmark. Long-context speedup is lower because paged attention
becomes a larger fraction of TPOT.

## System and measurement setup

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8,151 MiB |
| Driver | 610.62 |
| CUDA compiler | 12.8.93 |
| Rust | 1.97.1 |
| Checkpoint | `models/LFM2.5-1.2B-Instruct/model.safetensors`, 2,340,697,936 bytes |
| Runtime dtype | BF16 storage/output, FP32 sensitive math/FP8 GEMM compute |
| KV policy | PS16; K/V remain BF16 |
| Precision control | BF16 default; `--fp8-policy` startup opt-in |

Laptop clocks were not locked. Performance therefore uses same-process,
order-balanced interleaving, not two sequential groups.

## Numerical design

For a tensor with chosen finite clipping bound `c`, the tensor-wide E4M3 scale
is

```text
quantize_multiplier   s = 448 / c
dequantize_multiplier d = c / 448
q(x) = E4M3(clip(x, -c, c) * s)
```

The FP8 linear computes an FP32-accumulated GEMM with BF16 output:

```text
y_hat = (d_x * d_w) * (q(x) @ q(W)^T)
```

The implementation supplies `d_x * d_w` through cuBLASLt alpha while cached
tensor-wide scale pointers stay constant. This permits per-site calibrated
scales without allocating or uploading scale values in the decode path.
E4M3/E5M2 and their intended deep-learning use are defined in the
[FP8 formats paper](https://arxiv.org/abs/2209.05433); NVIDIA documents scalar
FP32 tensor-wide FP8 scaling in the
[cuBLASLt scaling modes](https://docs.nvidia.com/cuda/cublas/index.html#d-scaling-factors).

The local metrics are

```text
NRMSE = RMS(y_hat - y) / RMS(y)
cosine = <y_hat,y> / (||y_hat|| ||y||)
KL = sum_i p_BF16(i) log(p_BF16(i) / p_FP8(i))
PPL = exp(mean NLL)
```

Local GEMM similarity is only a screen. Promotion is determined by accumulated
hidden-state error, held-out NLL/KL/ranking metrics and measured E2E TPOT.

## Phase 1B: calibration

The calibration source is WikiText-2 train; valid was used for policy selection
and test only for final validation. WikiText retains case, punctuation and full
articles, as described by the
[dataset authors](https://www.salesforce.com/blog/the-wikitext-long-term-dependency-language-modeling-dataset).
The exact files came from the PyTorch Examples WikiText-2 mirror.

| Split | SHA-256 |
|---|---|
| train | `9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f` |
| valid | `f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8` |
| test | `d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0` |

Measured calibration coverage:

| Metric | Result |
|---|---:|
| Sequences / tokens | 256 / 75,936 |
| Length range | 96--768 |
| Length distribution | 25% x96, 35% x192, 25% x384, 15% x768 |
| Prefill forwards | 256 |
| Teacher-forced decode M=1 forwards | 2,048 |
| Decode context range | 2--768 |
| Weight / prefill / decode sites | 77 / 65 / 65 |
| Minimum observations per prefill/decode site | 256 / 2,048 |
| Non-finite values | 0 |

The strongest measured outliers were prefill `layers.7.mlp.down.input`
(`amax/p99.99=108.65`) and decode `layers.15.conv.output.input` (38.10),
`layers.9.mlp.down.input` (23.40), and `layers.7.mlp.down.input` (19.14).
The complete table is in `fp8/calibration-outliers.json` and the method is in
`fp8/calibration.md`.

## Phase 2A: real-checkpoint GEMM error

All 77 GEMM sites were tested with 64 reservoir-sampled real decode activations
and nine activation/weight scale pairs from `amax`, `p99.99`, and `p99.9`.

| Metric over each site's selected local trial | Result |
|---|---:|
| Selected strategy | amax/amax at 77/77 sites |
| NRMSE range | 0.01455--0.04422 |
| Minimum cosine | 0.999022 |
| Non-finite outputs across all trials | 0 |
| Sites passing cosine >=0.995 and NRMSE <=0.10 | 77/77 |

Percentile scales were evaluated, not assumed inferior. For this checkpoint,
their clipping error made every selected minimum worse than amax/amax. Passing
this local screen did not mean that all sites were enabled.

## Phase 2B/2C: propagation and policy search

Single-site validation captured 33 streaming points: mixer residual and FFN
residual for 16 layers, plus final RMSNorm. The most sensitive sites were:

| Site | Local NRMSE | Final hidden NRMSE | Mean logit KL | Sensitivity score |
|---|---:|---:|---:|---:|
| `layers.0.mlp.gate_up` | 0.02588 | 0.08879 | 0.02390 | 0.11269 |
| `layers.0.mlp.down` | 0.02597 | 0.08813 | 0.01009 | 0.10759 |
| `layers.1.mlp.gate_up` | 0.02406 | 0.08614 | 0.01103 | 0.10017 |
| `layers.1.conv.input` | 0.02019 | 0.07181 | 0.01494 | 0.08675 |
| `layers.1.mlp.down` | 0.02539 | 0.06804 | 0.00928 | 0.08566 |

This demonstrates why local error alone is insufficient: layer-0 sites with
about 2.6% local NRMSE produced roughly 8.8% final hidden NRMSE.

The greedy search ranked expected latency saving divided by sensitivity, tried
all 77 sites, accepted 50 through its fast gate and rolled back 27. Full
held-out evaluation then tested standard policies and frontier sizes
1/2/4/8/12/16/24/32/50. Frontier 24 failed because peak hidden NRMSE reached
0.12077; frontier 16 stayed below the 0.10 propagation gate (peak 0.09120).

The final policy contains 16 sites, all tensor-wide E4M3 with amax/amax:

```text
Gate/Up: layers 2, 3, 5, 7, 8, 9, 11, 15
Down:    layers 6, 8, 9, 10, 12, 14, 15
LM head
```

Conv and attention projections remain BF16. BF16 weights also remain resident
for prefill/reference/fallback. The 16 persistent FP8 copies occupy an
implementation-derived 520,093,696 bytes (496 MiB): eight 16384x2048 Gate/Up
matrices, seven 2048x8192 Down matrices and one 65536x2048 LM head. This is a
deliberate first-version memory cost, not an allocator measurement.

## Phase 3: model quality

Policy selection used the disjoint WikiText-2 valid split: 16 sequences, 1,872
next-token observations. `policy_frontier_16` passed with relative NLL delta
+0.1881%, mean KL 0.01093, peak hidden NRMSE 0.09120 and minimum hidden cosine
0.99584.

The frozen policy was then evaluated once on the untouched test split:

| Independent test metric | BF16 / FP8 result |
|---|---:|
| Sequences / source tokens / observations | 32 / 5,696 / 5,664 |
| Mean NLL | 3.93133 / 3.93980 |
| Absolute / relative NLL delta | +0.00847 / +0.2154% |
| Perplexity | 50.9747 / 51.4082 |
| Perplexity delta | +0.4335 |
| KL mean / p50 / p95 / p99 | 0.01120 / 0.00795 / 0.03160 / 0.06448 |
| Mean logit cosine | 0.998010 |
| Top-1 agreement | 92.96% |
| Mean top-5 / top-10 overlap | 93.63% / 94.02% |
| Final RMSNorm NRMSE / cosine | 0.08633 / 0.99627 |
| Non-finite logits/hidden values | 0 / 0 |
| Quality gate | PASS |

Greedy diagnostic: 8 fixed prompts, temperature 0, 32 token cap. Exact sequence
agreement was 5/8 (62.5%), output-length agreement 8/8. Divergent prompts first
diverged at tokens 16, 27 and 4. The fixed factual, Rust, identity, GPU and
Fibonacci prompts agreed exactly; no output was non-finite or degenerate. This
is diagnostic rather than the hard gate because a small rank-1/rank-2 change
causes an autoregressive trajectory split. The teacher-forced NLL/KL and hidden
metrics remain the primary evidence.

## Phase 4: production path

The runtime implementation has these invariants:

- FP8 weights are quantized once when the startup policy is installed.
- The policy JSON is resolved once into typed `Option<Fp8LinearWeight>` fields;
  decode performs no site-name formatting or hash lookup.
- FP8 dispatch requires `M=1` and an existing cache position. Initial prefill,
  including a one-token prefill, remains BF16.
- Activation E4M3 tensors use the bounded typed temporary pool. Stable measured
  workloads recorded 203,200 FP8 decode hits and zero misses across the five
  FP8 arms (40,640 hits per context). BF16 decode temporaries also recorded
  2,298,700 hits and zero misses across those arms.
- LM-head FP8 uses a persistent quantized copy of the tied BF16 embedding and a
  zero-copy last-hidden-row device view.
- KV, softmax, RMSNorm, RoPE, residual, embedding and sampling precision are
  unchanged. `launch.rs` and frozen kernels were not changed.

Primary implementation locations:

| Concern | Modules |
|---|---|
| Calibration and statistics | `src/model/calibration.rs`, `src/engine/runner.rs` |
| GEMM error and propagation | `src/model/fp8_analysis.rs`, `src/model/evaluation.rs` |
| Policy and typed weights | `src/model/quantization.rs`, `src/model/lfm2.rs` |
| FP8 GEMM/scaling | `src/cuda/blaslt/mod.rs`, `src/cuda/blaslt/fp8.rs` |
| Quantization and pooling | `src/cuda/kernels/fp8_quantize.rs`, `src/cuda/runtime.rs`, `src/ops/linear.rs` |
| CLI/offline workflows | `src/main.rs`, `src/engine/runner.rs` |

## Phase 5: interleaved E2E performance

Each workload used PS16, exactly 128 decode steps, two BF16/FP8 warm-up pairs
and 20 measured pairs. Pair order alternated between BF16-first and FP8-first.
The fixed-step benchmark ignores EOS only inside benchmark mode so every arm
executes the same number of model forwards; production generation still stops
at EOS.

| Context | BF16 TPOT mean / p50 / p95 | FP8 TPOT mean / p50 / p95 | Paired speedup mean / p50 / p95 | Pair range |
|---:|---:|---:|---:|---:|
| 40 | 7.675 / 7.773 / 8.067 ms | 6.286 / 6.318 / 6.607 ms | 1.221 / 1.221 / 1.236x | 1.191--1.318x |
| 138 | 8.015 / 7.976 / 8.222 ms | 6.638 / 6.638 / 6.813 ms | 1.207 / 1.210 / 1.232x | 1.185--1.232x |
| 516 | 8.373 / 8.373 / 8.422 ms | 7.016 / 7.007 / 7.058 ms | 1.193 / 1.194 / 1.200x | 1.181--1.208x |
| 2,056 | 10.902 / 11.208 / 11.413 ms | 9.419 / 9.538 / 9.668 ms | 1.157 / 1.175 / 1.195x | 1.064--1.215x |
| 8,202 | 22.023 / 22.021 / 22.120 ms | 20.282 / 20.298 / 20.385 ms | 1.086 / 1.086 / 1.091x | 1.078--1.091x |

Short-context TTFT was 8.811 ms BF16 versus 8.848 ms FP8, as expected from a
BF16 prefill. At 8,202 tokens it was 2,686.1 versus 2,686.0 ms. Full-request
BF16-pool misses at long contexts came entirely before the decode boundary;
decode-specific BF16 and FP8 miss counters were zero in every measured arm.
The decreasing TPOT speedup is consistent with an Amdahl decomposition: the
selected weight GEMMs shrink while paged XQA/KV traffic grows with context.

## Phase 6: post-FP8 profile

Detailed CUDA-event profiling used the same 13-token prompt, 60 generated
tokens before EOS, four skipped steps and 48 measured decode steps. Detailed
events add overhead, so this table ranks bottlenecks and is not the promotion
latency baseline.

| Region | BF16 | FP8 | Region speedup |
|---|---:|---:|---:|
| Decode GPU envelope | 8,483.9 us | 7,060.2 us | 1.202x |
| MLP Gate/Up GEMM | 3,082.2 us | 2,376.2 us | 1.297x |
| MLP Down GEMM | 1,614.0 us | 1,369.9 us | 1.178x |
| MLP total | 4,766.3 us | 3,827.8 us | 1.245x |
| LM head | 903.0 us | 379.4 us | 2.380x |
| Conv total | 1,192.3 us | 1,134.5 us | 1.051x |
| Attention total | 755.1 us | 814.9 us | 0.927x |

The absolute gains are concentrated exactly where the policy enabled FP8. Conv
and attention now account for a larger relative share; the single profile run
does not prove their small absolute differences are regressions. Further
fusion or a new kernel should require an interleaved component benchmark.

## Phase 7: regression result

Final gates:

```text
cargo fmt --check       PASS
cargo check --release   PASS, zero warnings
cargo test --release    PASS: 54 passed, 0 failed, 11 ignored benchmarks
```

The suite includes deterministic BF16-to-E4M3 quantization, FP8 cuBLASLt GEMM
correctness/error bounds, BF16 reference operators, PS16/PS32 KV and attention,
and policy/statistics metric tests. GPU tests were executed with device access;
no regression was suppressed or reclassified as ignored.

## Decision and next work

`PROMOTE` means the selected policy is ready for opt-in production use on the
measured checkpoint/GPU and satisfies the master milestone. It does not mean
turning FP8 on silently for all hardware. BF16 remains the default and can be
selected simply by omitting `--fp8-policy`.

Future work, outside this milestone:

1. Batch/continuous scheduling and MXFP8 at larger M, where the earlier
   feasibility data was stronger.
2. A separate FP8-prefill study; the current evidence is decode-M=1 only.
3. Long-context paged XQA optimization, because FP8's Amdahl benefit falls to
   1.089x by context 8,202.
4. Reduce the 496 MiB duplicated FP8-weight cost only after a memory-pressure
   benchmark proves it necessary.
5. FP4 or speculative decoding only as independent, gated milestones.

Machine-readable evidence is under `docs/fp8/`: calibration, outliers, GEMM
trials, sensitivity, search trace, policy frontier quality, frozen selected
policy, independent final quality, interleaved E2E data and BF16/FP8 profiles.
