__kernel void matmul(
  __global float* a,
  __global float* b,
  __global float* out,
  const unsigned int N)
{
  int idx = get_global_id(0);
  int row = idx / N;
  int col = idx % N;
  float sum = 0.0;
  for (int i = 0; i < N; i++) {
    sum += a[row*N + i] * b[i*N + col];
  }
  out[row*N + col] = sum;
}
