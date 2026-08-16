test cuda::kernels::tests::rms_norm::bench_rms_norm_bf16 ... rows=    1 | mean=   9.144 us | p50=   7.588 us | p95=  14.761 us | min=   5.806 us | logical=    1.34 GB/s | physical=    1.79GB/s |       109366 rows/s
rows=    4 | mean=   7.732 us | p50=   7.053 us | p95=  12.330us | min=   6.181 us | logical=    6.36 GB/s | physical=    8.48 GB/s |       517355 rows/s
rows=   16 | mean=   7.483 us | p50=   7.150 us | p95=  11.129us | min=   6.137 us | logical=   26.28 GB/s | physical=   35.03 GB/s |      2138265 rows/s
rows=   64 | mean=   8.293 us | p50=   7.347 us | p95=  15.266us | min=   6.357 us | logical=   94.83 GB/s | physical=  126.44 GB/s |      7717532 rows/s
rows=  256 | mean=  10.731 us | p50=  10.344 us | p95=  13.042us | min=   8.522 us | logical=  293.14 GB/s | physical=  390.85 GB/s |     23855759 rows/s
rows= 1024 | mean=  30.617 us | p50=  29.901 us | p95=  38.873us | min=  24.642 us | logical=  410.97 GB/s | physical=  547.96 GB/s |     33444928 rows/s
rows= 4096 | mean= 236.892 us | p50= 193.870 us | p95= 440.654us | min= 103.413 us | logical=  212.47 GB/s | physical=  283.29 GB/s |     17290551 rows/s