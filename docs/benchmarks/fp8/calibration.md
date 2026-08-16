# FP8 calibration record

This is the measured Phase 1B record for `LFM2.5-1.2B-Instruct`. The source was
the WikiText-2 train split; the validation and test splits were kept disjoint.

## Workload and coverage

| Item | Measured value |
|---|---:|
| Sequences | 256 |
| Model tokens | 75,936 |
| Sequence length | 96--768 |
| Prefill calls | 256 |
| Teacher-forced decode M=1 calls | 2,048 |
| Decode context range | 2--768 |
| Weight sites | 77, one observation each |
| Prefill activation sites | 65, 256 observations each |
| Decode activation sites | 65, 2,048 observations each |
| Non-finite values | 0 |

The deterministic 20-sequence length cycle is 5x96, 7x192, 5x384 and 3x768
tokens. It implements the requested 25%/35%/25%/15% length distribution.
Prefill statistics come from a full contiguous forward. Decode statistics come
from eight M=1 positions per sequence with a fresh KV cache and BF16
teacher-forced history.

## Largest activation outliers

The ratio is `amax / p99.99`; high values mean that an amax scale spends much
of E4M3's finite range on rare values, while a percentile scale introduces
clipping.

| Phase | Site | amax | p99.99 | ratio |
|---|---|---:|---:|---:|
| prefill | `layers.7.mlp.down.input` | 6.84375 | 0.06299 | 108.65 |
| prefill | `layers.15.conv.output.input` | 28.375 | 0.66797 | 42.48 |
| decode | `layers.15.conv.output.input` | 28.875 | 0.75781 | 38.10 |
| prefill | `layers.9.mlp.down.input` | 3.0625 | 0.12598 | 24.31 |
| decode | `layers.9.mlp.down.input` | 3.0625 | 0.13086 | 23.40 |
| decode | `layers.7.mlp.down.input` | 1.52344 | 0.07959 | 19.14 |

Outlier rank alone was not used to reject a site. Each scale was tested through
the real checkpoint GEMM and then through hidden/logit propagation. In this
checkpoint all 77 local minima selected amax/amax; percentile clipping did not
improve NRMSE enough to win.

## Artifacts

- `calibration-summary.json`: complete exact-BF16 histogram statistics.
- `calibration-outliers.json`: top 40 activation sites by `amax / p99.99`.
- `gemm-error.json`: nine real-input scale combinations per GEMM site.
- `sensitivity.json`: one-site-at-a-time downstream propagation.
- `policy-search.json`: cumulative accept/rollback decisions.

The corpus is a calibration/evaluation proxy, not evidence of general task
accuracy. The final independent test-split quality result is recorded in
`quality-final-test.json` and summarized in `../fp8_report.md`.
