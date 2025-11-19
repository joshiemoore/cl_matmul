#include <cuda_runtime.h>
#include <stdio.h>

#define TILE 16

__global__ void matmul_cu(const float* A, const float* B, float* C, const int N) {
  __shared__ float A_tile[TILE][TILE+4];
  __shared__ float B_tile[TILE][TILE+4];

  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  const int tile_row = threadIdx.y;
  const int tile_col = threadIdx.x;

  float sum = 0.0f;

  const int num_tiles = N / TILE;
  for (int t = 0; t < num_tiles; t++) {
    const int tile_offs = TILE * t;
    const int tile_col_idx = tile_offs + tile_col;
    const int tile_row_idx = tile_offs + tile_row;
    A_tile[tile_row][tile_col] = (row < N && tile_col_idx < N) ? A[row*N + tile_col_idx] : 0.0f;
    B_tile[tile_col][tile_row] = (tile_row_idx < N && col < N) ? B[tile_row_idx*N + col] : 0.0f;
    __syncthreads();

    for (int k = 0; k < TILE; k++) {
      sum += A_tile[tile_row][k] * B_tile[tile_col][k];
    }
    __syncthreads();
  }

  if (row < N && col < N) {
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

  dim3 threadsPerBlock(TILE, TILE);
  // NB: this assumes N is a perfect multiple of tile size, which
  // is true here since we're just hardcoding N=4096
  dim3 blocksPerGrid(N/TILE, N/TILE);

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
