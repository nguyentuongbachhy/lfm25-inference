test cuda::kernels::tests::silu_mul::bench_silu_mul_bf16 ... block_size=256
N=      1024 | mean=   9.198 us | p50=   9.685 us | p95=  12.441 us | min=   5.685 us | logical=    0.67 GB/s |    111329208 elem/s
N=      4096 | mean=   7.534 us | p50=   6.873 us | p95=  10.568 us | min=   5.454 us | logical=    3.26 GB/s |    543641457 elem/s
N=     16384 | mean=   7.651 us | p50=   6.799 us | p95=  11.091 us | min=   5.596 us | logical=   12.85 GB/s |   2141297166 elem/s
N=     65536 | mean=   8.824 us | p50=   7.511 us | p95=  12.616 us | min=   5.538 us | logical=   44.56 GB/s |   7426699398 elem/s
N=    262144 | mean=   9.071 us | p50=   8.487 us | p95=  14.023 us | min=   6.609 us | logical=  173.40 GB/s |  28899348295 elem/s
N=   1048576 | mean=  16.903 us | p50=  16.433 us | p95=  20.790 us | min=  12.592 us | logical=  372.21 GB/s |  62034875613 elem/s
N=   4194304 | mean=  75.292 us | p50=  70.813 us | p95=  99.169 us | min=  39.278 us | logical=  334.24 GB/s |  55706870617 elem/s
N=  16777216 | mean= 834.835 us | p50= 769.365 us | p95=1235.722 us | min= 602.268 us | logical=  120.58 GB/s |  20096456282 elem/s