#define TS 16

__kernel void matmul(
  __global float* A,
  __global float* B,
  __global float* out,
  const unsigned long int N)
{
  int col = get_global_id(0);
  int row = get_global_id(1);
  int col_local = get_local_id(0);
  int row_local = get_local_id(1);

  __local float Asub[TS][TS];
  __local float Bsub[TS][TS];

  float sum = 0.0;

  const int num_tiles = N / TS;
  for (int t = 0; t < num_tiles; t++) {
    const int tile_col = TS*t + col_local;
    const int tile_row = TS*t + row_local;
    Asub[row_local][col_local] = A[row*N + tile_col];
    Bsub[row_local][col_local] = B[tile_row*N + col];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < TS; k++) {
      sum += Asub[row_local][k] * Bsub[k][col_local];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }
  out[row*N + col] = sum;
}
