# NVFP4 SM120 research decision

## Final status

**REJECTED for production on the current LFM2.5-1.2B-Instruct checkpoint and RTX 5060 Laptop / SM120 target.**

The NVFP4 direction demonstrated strong primitive and isolated GEMM performance, but the final model-level confirmation did not retain any production-safe site. No NVFP4 runtime path is enabled in production and no research implementation is merged into `main`.

The retained production precision policy remains the validated selective E4M3 decode policy with BF16 fallback/prefill.

## Scope

- Model: LFM2.5-1.2B-Instruct
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, Blackwell SM120
- CUDA: 12.8.x
- Research branch: `agent/nvfp4-sm120`
- Production baseline before this study: `main` at `a3a5a63a20107874bd1dc2257f2f344f6a26d93e`
- NVFP4 scale recipe retained after Phase 2A: nearest UE4M3 block scale per 16 E2M1 values
- Round-up UE4M3 scaling: rejected

## Decision ledger

### Phase 0 / 0.5 - SM120 primitive and cache-controlled GEMM

Decision: **PASS as a performance primitive.**

CUTLASS 4.7 SM120 block-scaled NVFP4 showed a meaningful M=1 decode advantage over the existing E4M3 GEMM path under cache-controlled conditions. Representative cold-path speedups were approximately 1.5x to 1.65x depending on Gate/Up, Down, and LM-head shape.

This established that the format/kernel was performance-interesting, but it did not establish model quality.

### Phase 1 - persistent W4 + dynamic activation quantization

Decision: **PASS as an isolated decode implementation study.**

Representative cold-path E4M3 quant+GEMM to NVFP4 end-to-end improvements were:

| Site family | E4M3 | NVFP4 | Speedup |
|---|---:|---:|---:|
| Down M=1 | 66.828 us | 43.257 us | 1.545x |
| Gate/Up M=1 | 101.949 us | 66.905 us | 1.524x |
| LM head M=1 | 372.792 us | 225.378 us | 1.654x |

The external CUTLASS replay path used later for quality work was deliberately excluded from performance conclusions.

### Phase 1b - synthetic numerical sanity

Decision: **PASS only as packing/layout sanity.**

Synthetic real-shape studies had approximately 14-15% relative L2 error despite high cosine similarity. This was considered too large to justify production integration without real-checkpoint propagation studies.

### Phase 2A - real-checkpoint local characterization

Decision: **FILTER / nearest-only.**

Nearest UE4M3 scaling was retained. Round-up scaling was rejected because it worsened the LM-head local metrics, including NRMSE and KL.

LM head initially appeared to be the strongest local candidate under nearest scaling, while Down layers 6, 8, and 10 were classified as high-risk. Phase 2A was explicitly treated as a local screen rather than a production quality result.

### Phase 2B - sampled single-site propagation

Decision: **five single-site survivors, cumulative frontier reduced to two sites.**

Single-site screening retained only:

- `layers.5.mlp.gate_up`
- `layers.8.mlp.gate_up`
- `layers.9.mlp.gate_up`
- `layers.11.mlp.gate_up`
- `layers.15.mlp.gate_up`

The bounded cumulative policy search retained only:

- `layers.8.mlp.gate_up`
- `layers.9.mlp.gate_up`

LM head was rejected at model level despite its strong local Phase-2A metric. The other single-site survivors were not pursued further because adding them to the cumulative frontier violated the propagation screen.

Phase 2B was a sampled screen only and therefore did not authorize a production backend.

### Phase 2C - disjoint test confirmation

Decision: **REJECT NVFP4.**

Phase 2C tested the current validated 16-site production E4M3 policy against focused NVFP4 replacement of layer 8, layer 9, and layers 8+9 on the disjoint WikiText-2 test split.

| Policy | relNLL | mean KL | KL p95 | logit cosine | top1 | final hidden NRMSE | final hidden cosine | Screen |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| E4M3 baseline | -0.014262 | 0.013921 | 0.064480 | 0.997419 | 0.9219 | 0.088714 | 0.996058 | baseline |
| NVFP4 layer 8 | -0.001586 | 0.030752 | 0.073695 | 0.996335 | 0.9219 | 0.109982 | 0.993940 | REJECT |
| NVFP4 layer 9 | -0.015335 | 0.020064 | 0.076412 | 0.996709 | 0.9062 | 0.104399 | 0.994541 | REJECT |
| NVFP4 layers 8+9 | -0.008066 | 0.040615 | 0.098404 | 0.995425 | 0.9219 | 0.115170 | 0.993349 | REJECT |

The production-candidate hidden-state screen was:

```text
final_hidden_nrmse <= 0.10
final_hidden_cosine >= 0.995
no non-finite values
```

All focused NVFP4 candidates failed the hidden-state requirement on the disjoint test confirmation. The 8+9 candidate also increased mean KL by approximately 0.026693 relative to the E4M3 sampled baseline and shifted relative NLL by +0.006196 relative to that baseline.

No single-site candidate survived Phase 2C. Under the predeclared stop condition this terminates the direction before any in-process production backend work.

## Why the direction is rejected

The rejection is not because NVFP4 is slow. The isolated SM120 kernel evidence was favorable.

The rejection is because the format's additional quantization error did not remain inside the model-level propagation envelope when confirmed on a disjoint split. The remaining performance opportunity therefore does not justify adding a second production low-precision backend, persistent W4 storage, runtime dispatch state, quantization code, maintenance cost, and regression surface.

This is a model-quality-limited rejection.

## Production decision

Do not implement or merge:

- production NVFP4 weight storage;
- production NVFP4 activation quantization;
- NVFP4 decode precision dispatch;
- CUTLASS NVFP4 runtime dependency;
- NVFP4-specific fusions.

Keep:

- BF16 reference/fallback;
- BF16 prefill;
- current validated selective E4M3 decode policy;
- existing production runtime at the validated frontier.

The complete experimental harness and detailed evidence remain archived on `agent/nvfp4-sm120` and are intentionally not merged into `main`.

## Revisit conditions

Do not reopen this exact direction through additional layer-by-layer tuning. Revisit NVFP4 only if at least one material premise changes, for example:

- a materially different SM120/SM121 production kernel or library path;
- a different quantization/scaling method with new model-level evidence;
- a different checkpoint whose sensitivity profile is materially better;
- hardware where the performance gap versus E4M3 changes enough to justify a different quality/complexity tradeoff.

Nearest-vs-round-up tuning, LM-head rescue, Down 6/8/10 rescue, and expansion beyond the Phase-2B frontier are closed for this experiment.

## Next optimization phase

Start the next branch from the updated `main`, not from `agent/nvfp4-sm120`.

The next planned direction is FP8 KV cache unless a fresh production profile shows a higher-value bottleneck. Before implementation, re-profile the current main and quantify the maximum plausible E2E gain using Amdahl's law.
