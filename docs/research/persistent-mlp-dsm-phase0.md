# Persistent MLP / DSM — Phase 0

## Baseline

This direction starts from `main` after the bounded CUDA Graph long-context extension was promoted.

Closed directions remain closed: custom standalone tiny-M GEMM, packed QKV, FP8 KV, W8A8/W8A16, MXFP8, NVFP4, RMSNorm-to-FP8 fusion, and cuBLASLt autotuning are not reopened.

## Motivation

The post-FP8 profile attributes about 3.8 ms of a roughly 7 ms B1 decode step to the MLP region. Small launch or boundary optimizations have repeatedly produced large primitive wins but <=1% end-to-end gains.

A persistent MLP is only interesting if it removes a material intermediate-memory boundary, not if it simply replaces cuBLASLt with another standalone GEMM.

The target operator group is:

```text
RMSNorm -> Gate/Up -> SwiGLU -> Down -> Residual
```

For B1, the hidden vector is 2048 BF16 elements (4 KiB) and the activated intermediate is 8192 elements: 16 KiB in BF16 or 8 KiB in E4M3. These sizes fit comfortably inside SM120 shared-memory capacity.

CUDA compute capability 12.x supports Thread Block Clusters and Distributed Shared Memory (DSM). Blocks in one cluster are co-scheduled in one GPC and may access each other's shared memory. This provides a possible producer/consumer handoff for activation tiles without first materializing them to HBM.

## Phase 0 question

Before implementing Tensor Core MLP math, prove that the target RTX 5060 Laptop / SM120 can launch an 8-block cluster and that remote DSM handoff is competitive with a global-scratch handoff for activation-sized tiles.

This phase changes no model code.

## Microbenchmark

Test-only CUDA module `dsm_handoff.cu` contains two kernels with the same compile-time cluster size of 8 blocks.

Reference:

```text
producer block 0
    input -> global scratch
cluster.sync
consumer blocks 1..7
    global scratch -> output
```

Candidate:

```text
producer block 0
    input -> local shared memory
cluster.sync
consumer blocks 1..7
    producer shared memory through DSM -> output
```

Each consumer writes one complete copy of the source tile. The output therefore gives a simple bit-exact correctness check.

Tile sizes:

- 4096 BF16 elements = 8 KiB, representative of an E4M3/BF16-sized partial activation boundary;
- 8192 BF16 elements = 16 KiB, the complete BF16 activated intermediate for B1.

Use balanced paired GPU timing in one process to reduce laptop clock and thermal bias.

## Phase 0 gate

Continue to Tensor Core / persistent-MLP Phase 1 only if:

- both tile sizes are bit-exact;
- the 8-block cluster launches successfully on SM120;
- DSM has no material p95 regression at either tile size;
- at least one tile has mean speedup >=1.15x versus global scratch;
- the other tile is not worse than 1.05x in mean latency.

If DSM is neutral or slower, reject this producer/consumer architecture before implementing model math.

## Phase 1 concept if Phase 0 passes

Do not build a standalone replacement GEMM. Use a cluster-resident operator group.

A plausible first topology is:

```text
cluster
  producer/compute warps:
      load x tile
      Gate/Up tensor-core work
      SwiGLU
      publish activation tile in DSM

  Down consumer blocks:
      read activation tile from DSM
      accumulate disjoint output-channel tiles

  cluster synchronization between activation tiles
```

The key invariant is that the full 8192-wide activated vector is not written to global memory between SwiGLU and Down.

Phase 1 must compare the complete MLP operator group against the current cuBLASLt + fused SwiGLU + cuBLASLt path. A custom GEMM primitive by itself is not sufficient evidence.

## Amdahl target

With the MLP region near 54% of the short-context decode envelope, a 1.10x complete-MLP speedup implies roughly a 1.05x whole-step ceiling, and a 1.20x MLP speedup implies roughly a 1.10x whole-step ceiling. This is finally large enough to justify architectural complexity if the operator-group benchmark supports it.
