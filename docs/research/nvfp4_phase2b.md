# NVFP4 Phase 2B: sampled single-site propagation

## Status

**Complete on the RTX 5060 Laptop host. No production promotion or `main` merge.**

Phase 2A established that nearest UE4M3 block scales are the only retained
NVFP4 recipe. Round-up scaling is closed as rejected. Phase 2B asked whether
the actual model tolerates exact W4A4 NVFP4 output at any Phase-2A candidate
decode site once that error is propagated through the remaining network.

The host run completed successfully with the exact CUTLASS SM120 external
replay backend. The replay process is a quality harness only; its wall time is
not performance evidence.

## Why exact replay instead of production integration

The existing FP8 evaluator already implements teacher-forced independent
BF16/candidate caches, hidden-state capture, one-site sensitivity, NLL/KL
metrics, and a bounded cumulative policy search. Reimplementing those metrics
would introduce a second evaluation path.

Phase 2B therefore reuses that evaluator and intercepts only selected M=1
linear sites. At sampled decode positions the BF16 input and checkpoint weight
are replayed through the already-validated CUTLASS SM120 NVFP4 operator. The
BF16 result is uploaded back into the model and propagation continues normally.

All modifications to `src/` are injected into a detached temporary worktree by
`research/nvfp4/patch_phase2b_worktree.py`; the committed research branch keeps
production source unchanged.

## Candidate frontier

Phase-2A high-risk sites `down 6`, `down 8`, and `down 10` were excluded rather
than repeatedly attempting to rescue them.

The nearest-only Phase-2B frontier contained 13 sites:

- Gate/Up: 2, 3, 5, 7, 8, 9, 11, 15
- Down: 9, 12, 14, 15
- LM head

Measured Phase-1 latency savings used for ranking were:

| Family | E4M3 quant+GEMM | NVFP4 cold E2E | Expected saving/site |
|---|---:|---:|---:|
| Gate/Up M1 | 101.949 us | 66.905 us | 35.044 us |
| Down M1 | 66.828 us | 43.257 us | 23.571 us |
| LM head M1 | 372.792 us | 225.378 us | 147.414 us |

## Measured single-site propagation

`run_fp8_sensitivity` intentionally uses four sequences and four sampled decode
positions per sequence as a screening gate. The outer Phase-2B work package
loaded eight validation sequences, but the sensitivity report is therefore a
4 x 4 sampled proxy rather than final held-out quality.

| Site | Final hidden NRMSE | Final hidden cosine | Mean KL | Relative NLL | Screen |
|---|---:|---:|---:|---:|---|
| gate/up 2 | 0.103908 | 0.994625 | 0.017833 | +2.0690% | reject |
| gate/up 3 | 0.137444 | 0.990609 | 0.024231 | +2.3221% | reject |
| gate/up 5 | 0.075356 | 0.997160 | 0.009270 | -1.3306% | pass |
| gate/up 7 | 0.104586 | 0.994546 | 0.018288 | -0.2118% | reject |
| gate/up 8 | 0.054985 | 0.998487 | 0.004578 | -1.2720% | pass |
| gate/up 9 | 0.072358 | 0.997385 | 0.006914 | -0.8295% | pass |
| gate/up 11 | 0.083795 | 0.996490 | 0.008817 | -0.2378% | pass |
| gate/up 15 | 0.096829 | 0.995306 | 0.007049 | -0.2892% | pass |
| down 9 | 0.108481 | 0.994118 | 0.013386 | +2.0249% | reject |
| down 12 | 0.091621 | 0.995814 | 0.011016 | +0.5630% | reject |
| down 14 | 0.084580 | 0.996419 | 0.004623 | +1.7014% | reject |
| down 15 | 0.103638 | 0.994630 | 0.007862 | +1.1109% | reject |
| LM head | 0.012362 | 0.999924 | 0.020446 | +0.9183% | reject |

Five MLP sites survived the single-site screen: gate/up layers 5, 8, 9, 11,
and 15. LM head is closed as rejected: it already exceeded the 0.5% relative
NLL screen, and the cumulative policy search rejected it again before any MLP
site was accepted.

## Bounded cumulative policy search

Surviving/ranked sites were evaluated cumulatively using:

```text
expected_decode_saving_us / (single_site_sensitivity_score + 1e-6)
```

Only two sites were accepted:

```text
layers.8.mlp.gate_up
layers.9.mlp.gate_up
```

The two-site sampled policy reported approximately:

```text
relative NLL delta  -2.2284%
mean KL              0.007974
final hidden NRMSE   0.072347
final hidden cosine  0.997384
```

Adding gate/up 5, 11, or 15 after 8+9 failed the hidden propagation gate. They
are therefore closed for this NVFP4 recipe rather than carried forward for
more tuning. All Down candidates and LM head are also closed.

## Decision

**Phase 2B PASS as a frontier-reduction gate, not as a production quality gate.**

The only policy allowed to continue is the two-site Gate/Up frontier at layers
8 and 9. Phase 2B does not authorize production integration because:

1. sensitivity/policy search is sampled rather than full held-out evaluation;
2. it compared sparse NVFP4 perturbations against BF16, not the final hybrid
   policy that retains the validated production E4M3 sites;
3. the replay backend is intentionally external and cannot provide E2E latency.

## Phase 2C

The next bounded gate compares the current validated 16-site production E4M3
policy against:

- E4M3 with gate/up 8 replaced by exact NVFP4 replay;
- E4M3 with gate/up 9 replaced by exact NVFP4 replay;
- E4M3 with gate/up 8 and 9 both replaced by exact NVFP4 replay.

It uses a disjoint test corpus, eight sequences, and eight sampled positions by
default. If the 8+9 hybrid fails but one single-site replacement survives,
Phase 2C narrows to that one site. If neither survives, NVFP4 is rejected
without building an in-process backend.

Only a surviving Phase-2C policy may proceed to an in-process backend, full
teacher-forced held-out quality, autoregressive traces, ABBA E2E performance,
regression, and a final production PROMOTE/PARTIAL-PROMOTE/REJECT decision.
