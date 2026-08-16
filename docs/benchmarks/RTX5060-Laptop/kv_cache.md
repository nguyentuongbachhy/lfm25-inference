# Paged KV cache writer — RTX 5060 Laptop

Measured with CUDA events on August 14, 2026. BF16 K/V, HND layout,
8 KV heads, head dimension 64. Allocations and transfers are outside the timed
region. Values are kernel execution time per invocation.

| Page size | Tokens | Mean (us) | p50 (us) | p95 (us) | Min (us) |
|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 15.276 | 16.944 | 23.027 | 7.304 |
| 16 | 4 | 10.642 | 9.949 | 20.307 | 7.441 |
| 16 | 16 | 10.065 | 9.823 | 18.671 | 7.608 |
| 16 | 64 | 12.288 | 9.956 | 21.441 | 7.401 |
| 16 | 256 | 9.877 | 9.858 | 12.420 | 7.495 |
| 16 | 1024 | 15.013 | 12.821 | 23.912 | 8.286 |
| 16 | 4096 | 21.650 | 20.571 | 29.797 | 18.494 |
| 32 | 1 | 13.187 | 10.126 | 22.265 | 6.821 |
| 32 | 4 | 15.966 | 15.804 | 27.302 | 5.605 |
| 32 | 16 | 9.848 | 9.501 | 14.109 | 7.107 |
| 32 | 64 | 10.559 | 9.820 | 19.779 | 7.555 |
| 32 | 256 | 15.844 | 19.459 | 21.981 | 7.541 |
| 32 | 1024 | 11.412 | 9.118 | 21.456 | 8.370 |
| 32 | 4096 | 23.859 | 20.639 | 29.576 | 20.492 |

These writer-only measurements do not establish the final page-size winner.
The default remains PS16 until decode-attention latency and fragmentation are
benchmarked together under representative workloads.
