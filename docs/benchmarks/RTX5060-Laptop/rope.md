test cuda::kernels::tests::rope::bench_rope_qk_bf16 ... block_size=256
tokens=    1 | mean=   8.288 us | p50=   7.058 us | p95=  13.998 us | min=   6.246 us | qk_io=    1.24 GB/s |       120653 tok/s
tokens=    4 | mean=   7.701 us | p50=   7.041 us | p95=  12.689 us | min=   5.535 us | qk_io=    5.32 GB/s |       519385 tok/s
tokens=   16 | mean=   7.589 us | p50=   7.017 us | p95=  10.939 us | min=   6.012 us | qk_io=   21.59 GB/s |      2108339 tok/s
tokens=   64 | mean=   8.614 us | p50=   7.418 us | p95=  15.825 us | min=   5.869 us | qk_io=   76.08 GB/s |      7429586 tok/s
tokens=  256 | mean=  10.229 us | p50=  10.005 us | p95=  12.471 us | min=   8.214 us | qk_io=  256.27 GB/s |     25026113 tok/s
tokens= 1024 | mean=  30.476 us | p50=  30.565 us | p95=  38.086 us | min=  24.622 us | qk_io=  344.07 GB/s |     33600256 tok/s
tokens= 4096 | mean= 276.113 us | p50= 228.746 us | p95= 537.374 us | min= 178.216 us | qk_io=  151.91 GB/s |     14834499 tok/s