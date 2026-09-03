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

For B1, the hidden vector is 2048 BF16 elements (4 KiB) and the activated intermediate is 8192 elements: 16 KiB in BF16 or 8 KiB in E4M3. These sizes fit inside SM120 shared-memory capacity.

CUDA compute capability 12.x supports Thread Block Clusters and Distributed Shared Memory (DSM). Blocks in one cluster are co-scheduled in one GPC and may access each other's shared memory. This provided a possible producer/consumer handoff for activation tiles without first materializing them to HBM.

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

- 4096 BF16 elements = 8 KiB;
- 8192 BF16 elements = 16 KiB, the complete BF16 activated intermediate for B1.

Use balanced paired GPU timing in one process to reduce laptop clock and thermal bias.

## Precommitted Phase 0 gate

Continue to Tensor Core / persistent-MLP Phase 1 only if:

- both tile sizes are bit-exact;
- the 8-block cluster launches successfully on SM120;
- DSM has no material p95 regression at either tile size;
- at least one tile has mean speedup >=1.15x versus global scratch;
- the other tile is not worse than 1.05x in mean latency.

If DSM is neutral or slower, reject this producer/consumer architecture before implementing model math.

## Measured result

RTX 5060 Laptop GPU / SM120:

| Tile | Global mean | DSM mean | Mean speedup | Global p95 | DSM p95 | Exact |
|---:|---:|---:|---:|---:|---:|---|
| 8 KiB / 4096 BF16 | 11.260 us | 12.068 us | 0.9527x | 15.348 us | 17.294 us | true |
| 16 KiB / 8192 BF16 | 14.073 us | 17.599 us | 0.7999x | 16.193 us | 18.633 us | true |

The 8-block cluster launches correctly and DSM is numerically exact, but it is slower than the global-scratch reference at both activation sizes. The complete 16 KiB B1 activation handoff is about 20% slower on mean latency.

## Decision

**REJECT** the cluster-DSM producer/consumer persistent-MLP architecture.

The precommitted performance gate fails at both sizes. There is no justification for implementing Gate/Up or Down Tensor Core math on top of this handoff mechanism.

The result also indicates that the existing global-memory path for a 8–16 KiB activation boundary is already cheap on this GPU, likely benefiting from cache locality. Replacing that boundary with remote shared-memory access plus cluster synchronization does not improve the economics.

This rejection is scoped to the DSM handoff architecture. It does not prove that every possible operator-group or megakernel design is slower. A future direction must use a materially different mechanism and must not simply add DSM around the same work.

## Stop condition

Stop this branch here. Do not implement persistent MLP Tensor Core math on `agent/persistent-mlp-dsm`, and do not merge the experimental DSM code into `main`.
