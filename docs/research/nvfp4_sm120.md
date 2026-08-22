# NVFP4 SM120 research

## Scope

This branch evaluates whether native NVFP4 block-scaled GEMM is worth integrating into the runtime on the RTX 5060 Laptop GPU (SM120).

This is a phase-zero primitive viability experiment. It does not change the production runtime and it does not claim checkpoint-level numerical safety.

The production baseline remains the validated `main` commit:

```text
a3a5a63a20107874bd1dc2257f2f344f6a26d93e
```

## Why CUTLASS

CUTLASS 4.7.0 contains the official SM120 NVFP4 to BF16 example under `examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu`.

The example requires CUDA 12.8 or newer for SM120 and uses native block-scaled Tensor Core MMA. CUTLASS 4.6 also added `tileN=8,16` support for SM120 block-scaled GEMMs, which is relevant to decode because the runtime GEMMs have very small `M` and very large output width.

## Transposed benchmark mapping

The runtime linear is conceptually:

```text
Y[M,N] = X[M,K] * W[N,K]^T
```

The research harness benchmarks the equivalent transpose:

```text
Y^T[N,M] = W[N,K] * X[M,K]^T
```

This maps the runtime's tiny decode `M` to CUTLASS's GEMM `N`, allowing the new narrow `tileN` kernels to be tested directly.

The CUTLASS output is changed to column-major. A column-major `[N,M]` result has the same physical ordering as the runtime's row-major `[M,N]` result, so no output transpose kernel is implied by this mapping.

The harness builds five CUTLASS variants:

```text
tileN = 8, 16, 32, 64, 128
```

and reports the fastest verified variant for each production shape.

## Shapes

MLP Down:

```text
M = 1, 2, 8, 16, 32, 64
N = 2048
K = 8192
```

MLP Gate/Up:

```text
M = 1, 2, 8, 16, 32, 64
N = 16384
K = 2048
```

LM head:

```text
M = 1, 2, 8, 16
N = 65536
K = 2048
```

## Baseline

The script reuses the existing ignored Rust benchmark `bench_lfm_narrow_precision_gemms` from validated `main` rather than duplicating BF16 or tensor-wide E4M3 benchmark code.

Relevant rows are extracted for:

```text
bf16
fp8_e4m3
activation_quantize
fp8_quantize_gemm
```

The CUTLASS phase measures GEMM-only NVFP4 latency. Persistent weight conversion is therefore outside the timing, matching the first screening question: does the NVFP4 GEMM primitive have enough headroom to justify integration work?

## Run

```bash
bash scripts/bench_nvfp4_sm120.sh
```

Optional controls:

```text
CUTLASS_REF=v4.7.0
NVFP4_ITERATIONS=100
NVFP4_BUILD_JOBS=<jobs>
NVFP4_WORK_DIR=<path>
```

The default working directory is under `target/nvfp4-sm120`, so the downloaded CUTLASS checkout, build tree, binaries, and logs remain outside version control.

## Output

The most useful lines are:

```text
mlp_down,...,bf16,...
mlp_down,...,fp8_e4m3,...
mlp_down,...,fp8_quantize_gemm,...

nvfp4_cutlass site=... tileN=... mean_us=... verification=pass
nvfp4_best site=... tileN=... mean_us=...
```

Full logs are written to:

```text
target/nvfp4-sm120/rust-baseline.log
target/nvfp4-sm120/nvfp4-cutlass.log
```

## Decision gate

This phase should be killed early if native NVFP4 GEMM does not materially beat the current tensor-wide E4M3 GEMM.

A useful first screen is:

```text
NVFP4 GEMM <= 0.85-0.90 * tensor-wide FP8 GEMM
    continue

NVFP4 GEMM approximately equal to tensor-wide FP8
    reject unless a later same-process test reveals a clear advantage

NVFP4 GEMM slower than tensor-wide FP8
    reject
```

Laptop clocks are not locked and the CUTLASS executable is not yet measured in the same process as the Rust baseline. Borderline results must therefore be treated as inconclusive, not promoted.

## If phase zero passes

Only after a clear GEMM win should the branch proceed to phase one:

1. implement or bind BF16 to NVFP4 activation quantization;
2. keep NVFP4 weights persistent and conversion outside decode timing;
3. benchmark quantization, GEMM, and quantization plus GEMM in one process;
4. compare BF16, tensor-wide E4M3, and NVFP4 on identical inputs;
5. measure relative L2, cosine, and max absolute error;
6. run real-checkpoint site sensitivity;
7. run long sampled-token traces across multiple prompts, contexts, and batch sizes;
8. run full-model order-balanced ABBA before any production merge.

Do not merge this research branch into `main` based on the phase-zero benchmark alone.

## Phase 2A: real-checkpoint local characterization

**Status: complete; no production promotion.** This phase used the same
WikiText-2 train calibration stream as the validated E4M3 program and the
disjoint WikiText-2 validation split. The corpus hashes were checked before
the run:

```text
train  9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f
valid  f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8
```

Hardware and software were RTX 5060 Laptop (SM120, 8 GiB), driver 610.62,
CUDA 12.8.93, and CUTLASS v4.7.0. The bridge collected 64 reservoir-sampled
real decode M=1 rows per site after 256 calibration sequences, then compared
the NVFP4 output with the runtime BF16 reference. The tested sites are exactly
the 16 validated E4M3 decode sites; no new precision sites were introduced.

### Nearest UE4M3 scale result

The initial recipe stores `UE4M3(round-to-nearest-even(amax / 6))` and rounds
the FP4 values through CUTLASS. Local metrics are a screening gate only.

| Site | NRMSE | Cosine | Screen |
|---|---:|---:|---|
| gate/up 2 | 0.101763 | 0.994847 | candidate |
| gate/up 3 | 0.113162 | 0.993625 | candidate |
| gate/up 5 | 0.105274 | 0.994478 | candidate |
| gate/up 7 | 0.108159 | 0.994144 | candidate |
| gate/up 8 | 0.109813 | 0.993976 | candidate |
| gate/up 9 | 0.106210 | 0.994371 | candidate |
| gate/up 11 | 0.113672 | 0.993531 | candidate |
| gate/up 15 | 0.100662 | 0.994993 | candidate |
| down 6 | 0.368463 | 0.930610 | high risk |
| down 8 | 0.236338 | 0.972082 | high risk |
| down 9 | 0.119963 | 0.993533 | candidate |
| down 10 | 0.165477 | 0.986309 | high risk |
| down 12 | 0.148553 | 0.988921 | candidate |
| down 14 | 0.137574 | 0.990493 | candidate |
| down 15 | 0.105411 | 0.994454 | candidate |
| LM head | 0.095655 | 0.995421 | strong |

All outputs were finite. For LM head, top-1 agreement was 90.625%, top-5
overlap 88.438%, top-10 overlap 91.094%, and mean KL 0.022646. This is not a
quality pass: it is the only strong local candidate and therefore the first
site that a future propagation/teacher-forced study must test.

### Scale-recipe comparison

The research harness now supports `NVFP4_CHECKPOINT_SCALE_MODE=nearest` or
`round_up`. `round_up` starts from the same requested `amax / 6` scale and
advances by one finite UE4M3 encoding only when nearest-even would under-scale
that value. It was evaluated on the identical checkpoint/corpora/row protocol,
not inferred from synthetic data.

`round_up` did not improve the decisive LM-head result:

| Recipe | NRMSE | Cosine | Top-1 | Mean KL |
|---|---:|---:|---:|---:|
| nearest | 0.095655 | 0.995421 | 90.625% | 0.022646 |
| round-up | 0.131277 | 0.992869 | 84.375% | 0.038500 |

It also made 14 of the 16 sites worse in NRMSE, and moved `down 12` from
candidate to high risk. The small improvements at `down 6` and `down 8` leave
both far outside the candidate threshold. **Decision: reject round-up;
retain nearest as the sole NVFP4 research recipe.**

### Reproduction

```bash
# nearest (default)
bash scripts/check_nvfp4_checkpoint.sh /tmp/wikitext-2-train.txt /tmp/wikitext-2-valid.txt

# controlled round-up comparison, separate ignored artefacts
NVFP4_WORK_DIR=target/nvfp4-sm120-roundup \
NVFP4_CHECKPOINT_SCALE_MODE=round_up \
  bash scripts/check_nvfp4_checkpoint.sh /tmp/wikitext-2-train.txt /tmp/wikitext-2-valid.txt
```

The logs are deliberately ignored under `target/nvfp4-sm120/` and
`target/nvfp4-sm120-roundup/`. Phase 2B remains required: enable only one
nearest-scale candidate at a time in a model-level research path and measure
hidden-state propagation, NLL/KL, ranking, and sampled traces before any
production integration.
