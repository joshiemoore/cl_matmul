#include <cuda_runtime.h>
#include <stdio.h>

#define RUNS 10

#define TILE 64
#define WPT 4

#if WPT == 2
  #define floatX float2
#elif WPT == 4
  #define floatX float4
#endif

__global__ void matmul_cu(const float* A, const float* B, float* C, const int N) {
  __shared__ float A_tile[TILE][TILE+4];
  __shared__ float B_tile[TILE][TILE+4];

  const int row = (blockDim.y * blockIdx.y + threadIdx.y) * WPT;
  const int col = (blockDim.x * blockIdx.x + threadIdx.x) * WPT;
  const int tile_row = threadIdx.y * WPT;
  const int tile_col = threadIdx.x * WPT;

  float sum[WPT][WPT] = { 0.0f };

  const int num_tiles = N / TILE;
  for (int t = 0; t < num_tiles; t++) {
    const int tile_offs = TILE * t;
    const int tile_col_idx = tile_offs + tile_col;
    const int tile_row_idx = tile_offs + tile_row;
    for (int br = 0; br < WPT; br++) {
      floatX A_v = *(floatX*)&A[(row+br)*N + tile_col_idx];
      floatX B_v = *(floatX*)&B[(tile_row_idx+br)*N + col];
      #pragma unroll
      for (int bc = 0; bc < WPT; bc++) {
        A_tile[tile_row+br][tile_col+bc] = reinterpret_cast<float*>(&A_v)[bc];
        B_tile[tile_col+bc][tile_row+br] = reinterpret_cast<float*>(&B_v)[bc];
      }
    }
    __syncthreads();

    for (int br = 0; br < WPT; br++) {
      for (int bc = 0; bc < WPT; bc++) {
        for (int k = 0; k < TILE; k++) {
          sum[br][bc] += A_tile[tile_row+br][k] * B_tile[tile_col+bc][k];
        }
      }
    }
    __syncthreads();
  }

  const int cidx = col / WPT;
  for (int br = 0; br < WPT; br++ ) {
    reinterpret_cast<floatX*>(C)[(row+br)*(N/WPT) + cidx] = reinterpret_cast<floatX*>(sum)[br];
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

  dim3 threadsPerBlock(TILE/WPT, TILE/WPT);
  // NB: this assumes N is a perfect multiple of tile size, which
  // is true here since we're just hardcoding N=4096
  dim3 blocksPerGrid(N/TILE, N/TILE);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  printf("running tests");
  float time_total;
  for (int i = 0; i < RUNS; i++) {
    cudaEventRecord(start);
    matmul_cu<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    cudaMemcpy(h_C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float run_ms = 0;
    cudaEventElapsedTime(&run_ms, start, stop);
    time_total += run_ms;
    printf(".");
    fflush(stdout);
  }
  float mean_time_ms = time_total / RUNS;
  printf("\n mean speed: %f GFLOP/s\n\n", FLOP / (mean_time_ms / 1000) / (float)1e9);

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
