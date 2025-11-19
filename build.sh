#!/usr/bin/env bash

gcc -O3 -o matmul matmul.c -lOpenCL
nvcc -o matmul_cu matmul.cu
