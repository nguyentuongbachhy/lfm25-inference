# Segmented & Ragged Tensor Core FlashAttention — Phase 0

## Baseline

BASE_BRANCH: agent/prefill-flash-attention
BASE_SHA: 4df5ffa0cb549cf84ef8f0b0cda19c9ba25c1bc2
RESEARCH_BRANCH: agent/ragged-flash-attention

Previous investigations established that:
- Contiguous single-sequence prefill was bound by warp-shuffle serialization and scalar math.
- Blackwell WMMA $16 \times 16 \times 16$ Tensor Core FlashAttention (`prefill_gqa_lfm2_bf16_flash`) solved this, accelerating 8K contiguous prefill by 2.52x.
- However, when multiple prefill sequences arrive simultaneously (`segment_offsets.len() > 2`), `BatchModelCache` currently falls back to `hybrid_ragged_attention_lfm2_bf16` (`hybrid_ragged_gqa_lfm2_bf16_body`), which processes tokens individually with scalar warp shuffles.

## Direction

DIRECTION: Segmented & Ragged Tensor Core FlashAttention
HYPOTHESIS: In multi-sequence batches (e.g. concurrent user prompts arriving together or chunked prefill), each sequence $s$ attends causally to its own tokens defined by `[segment_offsets[s], segment_offsets[s+1])`. By dispatching segmented sequences through tiled WMMA Tensor Core FlashAttention (with causal boundary masking constrained to each segment), warp-shuffle serialization stalls are eliminated, unlocking hardware tensor throughput for multi-sequence prefill.
AFFECTED_RUNTIME_FRACTION: 40%–75% of multi-sequence prefill execution time.
EXPECTED_LOCAL_SPEEDUP: 2.0x – 4.0x at the attention primitive layer.
AMDAHL_E2E_CEILING: 1.33x – 2.29x for multi-sequence prefill prompts.

## Precommitted Gates

PRIMITIVE_GATE: Paired speedup >= 1.25x across multi-sequence shapes (e.g. 2x512, 4x512, 2x2048)
NUMERICAL_GATE: Max NRMSE <= 0.05, min cosine >= 0.9999, 0 non-finite values (NaN/Inf)
MODEL_QUALITY_GATE: Logit cosine >= 0.9999, logit NRMSE <= 0.02, min hidden cosine >= 0.9998, 0 non-finite values
E2E_GATE: Full-model prompt TTFT paired speedup >= 1.10x for multi-sequence batches
P95_GATE: Paired p95 speedup >= 1.05x
STOP_CONDITION: Any NaN/Inf, or numerical gate failure, or speedup < 1.00x
ITERATION_BUDGET: 3 iterations

## Benchmark Protocol

- Balanced same-process ABBA GPU timing (`benchmark_gpu_paired`).
- Multi-sequence configurations:
  - 2 sequences x 512 tokens ($N=1024$)
  - 4 sequences x 512 tokens ($N=2048$)
  - 2 sequences x 2048 tokens ($N=4096$)
- Dtype: BF16 storage, FP32 accumulator/online softmax, BF16 output.
