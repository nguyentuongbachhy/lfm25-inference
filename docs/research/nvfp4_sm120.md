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
