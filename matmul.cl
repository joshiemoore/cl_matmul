__kernel void matmul(
  __global float* a,
  __global float* b,
  __global float* out,
  const unsigned long int N)
{
  int col = get_global_id(0);
  int row = get_global_id(1);
  float sum = 0.0;
  for (int i = 0; i < N; i++) {
    sum += a[row*N + i] * b[i*N + col];
  }
  out[row*N + col] = sum;
}
