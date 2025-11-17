# cl_matmul

multiplying 4096x4096 matrices with OpenCL, how many ~~GFLOP/s~~ TFLOP/s can we get on my RX580?

* [ec7a70e](https://github.com/joshiemoore/cl_matmul/commit/ec7a70ee9e5370b020b6e554cb1ff3c7ab626524): ~118 GFLOP/s - naive implementation
* [10cf4df](https://github.com/joshiemoore/cl_matmul/commit/10cf4df649e9f9e615108a25a714a6bc92151f11): ~420 GFLOP/s - optimized workgroup size
* [af25bcc](https://github.com/joshiemoore/cl_matmul/commit/af25bccd1387af8275ededbb19c4d0ed338c4f51): ~1.2 TFLOP/s - added local memory tiling

1.2 TFLOP/s is in about the same range as CLBlast's tuned GEMM kernels on this card, so I'll stop here for now.
