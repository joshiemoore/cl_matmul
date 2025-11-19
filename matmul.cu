#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matmul_cu(const float* A, const float* B, float* C, const int N) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < N && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
      sum += A[row*N + k] * B[k*N + col];
    }
    C[row*N + col] = sum;
  }
}

int main(void) {
  const int N = 4096;
  const unsigned long long int FLOP = N*N*(2ll*N);

  // host buffers
  float* h_A = (float*)calloc(N*N, sizeof(float));
  float* h_B = (float*)calloc(N*N, sizeof(float));
  float* h_C = (float*)calloc(N*N, sizeof(float));

  // fill input matrices with random values
  for (int r = 0; r < N; r++) {
    for (int c = 0; c < N; c++) {
      h_A[r*N + c] = rand() / (float)RAND_MAX;
      h_B[r*N + c] = rand() / (float)RAND_MAX;
    }
  }

  // device buffers
  float* d_A;
  float* d_B;
  float* d_C;
  cudaMalloc((void**)&d_A, N*N*sizeof(float));
  cudaMalloc((void**)&d_B, N*N*sizeof(float));
  cudaMalloc((void**)&d_C, N*N*sizeof(float));

  cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, N*N*sizeof(float), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid(
    (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
    (N + threadsPerBlock.y - 1) / threadsPerBlock.y
  );

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  matmul_cu<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
  cudaEventRecord(stop);

  cudaMemcpy(h_C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);
  float run_ms = 0;
  cudaEventElapsedTime(&run_ms, start, stop);

  printf("\nspeed: %f GFLOP/s\n\n", FLOP / (run_ms / 1000) / (float)1e9);

  // verify results
  unsigned int correct = 0;
  printf("verifying results...\n");
  for (int r = 0; r < N; r++) {
    for (int c = 0; c < N; c++) {
      float gpu_sum = h_C[r*N + c];
      float cpu_sum = 0.0f;
      for (int k = 0; k < N; k++) {
        cpu_sum += h_A[r*N + k] * h_B[k*N + c];
      }
      if (abs(cpu_sum - gpu_sum) < 0.001) {
        correct++;
      }
    }
  }
  printf("%d/%d correct results\n", correct, N*N);

  cudaFree(d_C);
  cudaFree(d_B);
  cudaFree(d_A);
  free(h_C);
  free(h_B);
  free(h_A);

  return 0;
}
