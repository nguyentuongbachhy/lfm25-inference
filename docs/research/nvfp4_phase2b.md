# NVFP4 Phase 2B: sampled single-site propagation

## Status

**Ready to run on the target host. No production promotion or `main` merge.**

Phase 2A established that nearest UE4M3 block scales are the only retained
NVFP4 recipe. Round-up scaling is closed as rejected. Phase 2B answers the
next question before any in-process/production backend is built:

> Does the actual model tolerate exact W4A4 NVFP4 output at any of the
> Phase-2A candidate decode sites once that error is propagated through the
> remaining network?

## Why exact replay instead of production integration

The existing FP8 evaluator already implements teacher-forced independent
BF16/candidate caches, hidden-state capture, one-site sensitivity, NLL/KL
metrics, and a bounded cumulative policy search. Reimplementing those metrics
would introduce a second evaluation path.

Phase 2B therefore reuses that evaluator and intercepts only selected M=1
linear sites. At sampled decode positions the BF16 input and checkpoint weight
are replayed through the already-validated CUTLASS SM120 NVFP4 operator. The
BF16 result is uploaded back into the model and propagation continues normally.

This backend is deliberately slow and uses CPU files plus an external process.
**Its wall-clock time is invalid for performance conclusions.** Its sole
purpose is exact model-quality screening before investing in a production
backend.

All modifications to `src/` are injected into a detached temporary worktree by
`research/nvfp4/patch_phase2b_worktree.py`; the committed research branch keeps
production source unchanged.

## Candidate frontier

Phase-2A high-risk sites `down 6`, `down 8`, and `down 10` are excluded rather
than repeatedly attempting to rescue them.

The nearest-only frontier contains 13 sites:

- Gate/Up: 2, 3, 5, 7, 8, 9, 11, 15
- Down: 9, 12, 14, 15
- LM head

The runner derives the carrier policy from
`docs/benchmarks/fp8/selected-policy.json` so existing E4M3 scale metadata does
not need to be duplicated. Those E4M3 values only initialize the normal decode
precision toggle; active candidate GEMMs are intercepted and replaced by exact
NVFP4 replay.

Measured Phase-1 latency savings used for ranking are:

| Family | E4M3 quant+GEMM | NVFP4 cold E2E | Expected saving/site |
|---|---:|---:|---:|
| Gate/Up M1 | 101.949 us | 66.905 us | 35.044 us |
| Down M1 | 66.828 us | 43.257 us | 23.571 us |
| LM head M1 | 372.792 us | 225.378 us | 147.414 us |

These measured savings rank quality survivors; replay-process latency is never
used.

## Single-site screen

Existing `run_fp8_sensitivity` is reused with four sequences and four sampled
decode positions per sequence. For every site it records:

- local Phase-2A NRMSE/cosine;
- final RMSNorm NRMSE/cosine;
- mean logit KL;
- relative NLL delta;
- expected measured decode saving;
- combined sensitivity score.

An MLP site remains viable only when:

```text
final hidden NRMSE <= 0.10
final hidden cosine >= 0.995
mean KL <= 0.05
relative NLL delta <= 0.5%
no numerical failure
```

LM head is special because it is after `final_rms_norm`. Its candidate run must
leave final hidden state effectively identical:

```text
final hidden NRMSE <= 1e-7
final hidden cosine >= 0.999999
```

and must also pass the KL/NLL screen above. A hidden-state change in an
LM-head-only trial indicates a harness bug, not acceptable propagation.

## Bounded policy search

Surviving sites are ranked by:

```text
expected_decode_saving_us / (single_site_sensitivity_score + 1e-6)
```

The cumulative search uses the existing bounded FP8 search machinery on two
sequences and four sampled positions. For Phase 2B the hidden gate is tightened
to NRMSE <= 0.10 and cosine >= 0.995.

This is a screening search, not final quality approval.

## Stop conditions

The phase deliberately prevents a rabbit hole:

1. If no single site survives, reject NVFP4 before building an in-process
   backend.
2. If only LM head survives, stop trying to force MLP sites into FP4. The next
   candidate is LM-head-only NVFP4.
3. If a small set survives, carry only that set into the in-process Phase-3
   backend. Do not expand to high-risk or untested sites.
4. Round-up scaling remains rejected and is not retested.

## Host-only execution

The runner refuses to execute unless the canonical host checkout and target
GPU are visible. CUDA/model work must not run in Codex sandbox/container.

```bash
cd /home/hyy4hc/source/lfm25-inference
git pull --ff-only
bash scripts/run_nvfp4_phase2b.sh /tmp/wikitext-2-valid.txt
```

Optional bounded controls:

```text
NVFP4_PHASE2B_SEQUENCES=8
NVFP4_PHASE2B_MAX_TOKENS=128
NVFP4_PHASE2B_WORK_DIR=target/nvfp4-sm120-phase2b
NVFP4_MODEL=models/LFM2.5-1.2B-Instruct
```

Outputs:

```text
target/nvfp4-sm120-phase2b/nvfp4-phase2b.json
target/nvfp4-sm120-phase2b/nvfp4-phase2b-summary.txt
```

The JSON is the source of truth. The text file is a concise decision ledger.

## Next decision

Phase 2B does not authorize a production merge. If at least one site survives,
the next work package is an in-process nearest-only NVFP4 backend for only that
frontier, followed by full held-out BF16/E4M3/NVFP4 quality, autoregressive
trace diagnostics, ABBA E2E performance, regression, and only then a
PROMOTE/PARTIAL-PROMOTE/REJECT decision for `main`.
