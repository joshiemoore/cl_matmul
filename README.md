# cl_matmul

## RX 580 - matmul.cl
multiplying 4096x4096 matrices with OpenCL, how many ~~GFLOP/s~~ TFLOP/s can we get on my RX 580?

* [ec7a70e](https://github.com/joshiemoore/cl_matmul/commit/ec7a70ee9e5370b020b6e554cb1ff3c7ab626524): ~118 GFLOP/s - naive implementation
* [10cf4df](https://github.com/joshiemoore/cl_matmul/commit/10cf4df649e9f9e615108a25a714a6bc92151f11): ~420 GFLOP/s - optimized workgroup size
* [af25bcc](https://github.com/joshiemoore/cl_matmul/commit/af25bccd1387af8275ededbb19c4d0ed338c4f51): ~1.2 TFLOP/s - added local memory tiling

1.2 TFLOP/s is in about the same range as CLBlast's tuned GEMM kernels on this card, so I'll stop here for now.

## RTX 3060 - matmul.cu
I bought an RTX 3060, now how many TFLOP/s can we get with CUDA?

* [0e1382a](https://github.com/joshiemoore/cl_matmul/commit/0e1382a6e9a5539ce29da2fb77c982979bdb7059): ~800 GFLOP/s - naive implementation
* [96a7782](https://github.com/joshiemoore/cl_matmul/commit/96a77821f55c0b30320aa26516039ebcea6758ad): ~1.4 TFLOP/s - shared memory tiling
* [98fb354](https://github.com/joshiemoore/cl_matmul/commit/98fb354f366177624d9ccdc3b493cf73fdc4cde5): ~1.543 TFLOP/s - 2x2 register blocking
* [8d49aca](https://github.com/joshiemoore/cl_matmul/commit/8d49acab1944a60970c64e4fb16b59101a73504e): ~1.7 TFLOP/s - transpose B on host for coalesced reads from global memory
* [9bdc9dd](https://github.com/joshiemoore/cl_matmul/commit/9bdc9dd0cedb9569592c92e1b13bde555b1ee560): ~1.8 TFLOP/s - vectorize loads from A
