# NVFP4 Phase 2C: focused hybrid confirmation

## Objective

Phase 2B reduced the NVFP4 frontier to two Gate/Up sites:

```text
layers.8.mlp.gate_up
layers.9.mlp.gate_up
```

Phase 2C answers one narrow question before an in-process backend is built:

> Does replacing the production E4M3 implementation at those sites with the
> exact nearest-scale NVFP4 result remain numerically acceptable when the
> rest of the validated 16-site E4M3 policy stays enabled?

This prevents spending implementation effort on a two-site policy that only
looked safe relative to an otherwise-BF16 model.

## Compared policies

All modes install `docs/benchmarks/fp8/selected-policy.json` first.
`NVFP4_REPLAY_ALLOWLIST` controls which enabled sites are intercepted by the
quality-only CUTLASS replay backend:

```text
e4m3   : no NVFP4 replay; current 16-site E4M3 baseline
l8     : E4M3 baseline with gate/up 8 replaced by NVFP4
l9     : E4M3 baseline with gate/up 9 replaced by NVFP4
l8-l9  : E4M3 baseline with gate/up 8 and 9 replaced by NVFP4
```

All other enabled sites remain on their normal production E4M3 path.

## Evaluation scope

The default confirmation uses:

```text
8 disjoint test sequences
max 256 source tokens
8 sampled decode positions per sequence
teacher-forced independent BF16/candidate caches
```

The test corpus must be disjoint from the Phase 2A/2B validation split. The
runner refuses the known validation SHA-256. An exact expected test SHA-256
can additionally be pinned with `NVFP4_PHASE2C_EXPECTED_SHA256`.

This remains a sampled quality gate. External replay wall time is explicitly
invalid for performance conclusions.

## Screen

A hybrid candidate must have no non-finites and satisfy:

```text
final hidden NRMSE <= 0.10
final hidden cosine >= 0.995
mean KL <= 0.020
relative NLL delta vs BF16 <= 0.75%
incremental relative NLL vs sampled E4M3 <= 0.50%
incremental mean KL vs sampled E4M3 <= 0.010
top-1 agreement no worse than sampled E4M3 by > 5 percentage points
```

These are bounded screening criteria, not final production quality gates.

## Decision tree

```text
8+9 passes
  -> proceed to an in-process two-site NVFP4 backend

8+9 fails, exactly one/both single replacements pass
  -> narrow to the best surviving single site
  -> proceed in-process only for that site

8+9 and both singles fail
  -> reject NVFP4
  -> no production backend
  -> no main merge
```

Phase-2B survivors 5/11/15 are deliberately not reopened. LM head and all Down
sites are also closed for this NVFP4 recipe.

## Host command

Use the same WikiText-2 test split used by the independent production E4M3
validation when available (`/tmp/wikitext-2-test.txt` in the recorded run):

```bash
cd /home/hyy4hc/source/lfm25-inference
git pull --ff-only
bash scripts/run_nvfp4_phase2c.sh /tmp/wikitext-2-test.txt
```

Optional reproducibility pin:

```bash
NVFP4_PHASE2C_EXPECTED_SHA256=<known-test-sha256> \
  bash scripts/run_nvfp4_phase2c.sh /tmp/wikitext-2-test.txt
```

Outputs are written under:

```text
target/nvfp4-sm120-phase2c/phase2c-e4m3.json
target/nvfp4-sm120-phase2c/phase2c-l8.json
target/nvfp4-sm120-phase2c/phase2c-l9.json
target/nvfp4-sm120-phase2c/phase2c-l8-l9.json
target/nvfp4-sm120-phase2c/nvfp4-phase2c-summary.txt
```

No production source is committed by this phase.
