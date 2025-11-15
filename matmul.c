#include <stdio.h>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#define N 4

void check_error(int err, const char* msg) {
  if (err != CL_SUCCESS) {
    printf("Error (%d): %s\n", err, msg);
    exit(1);
  }
}

int main() {
  int err;
  size_t global_len = N*N;

  cl_device_id device_id;
  cl_context ctx;
  cl_command_queue q_cmd;
  cl_program program;
  cl_kernel ko_matmul;

  // device I/O buffers
  cl_mem d_a;
  cl_mem d_b;
  cl_mem d_out;

  // host I/O buffers
  float* h_a = calloc(N*N, sizeof(float));
  float* h_b = calloc(N*N, sizeof(float));
  float* h_out = calloc(N*N, sizeof(float));

  // fill input matrices with random values
  for (int i = 0; i < global_len; i++) {
    h_a[i] = rand() / (float)RAND_MAX;
    h_b[i] = rand() / (float)RAND_MAX;
  }

  // setup platform and GPU device
  cl_uint num_platforms;
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  check_error(err, "getting num_platforms");
  if (num_platforms == 0) {
    printf("0 platforms available\n");
    return 1;
  }
  cl_platform_id platforms[num_platforms];
  err = clGetPlatformIDs(num_platforms, platforms, NULL);
  check_error(err, "getting platforms");
  for (int i = 0; i < num_platforms; i++) {
    err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
    if (err == CL_SUCCESS) {
      break;
    }
  }
  if (device_id == NULL) {
    check_error(err, "no device found");
    return 1;
  }

  ctx = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  check_error(err, "creating context");
  q_cmd = clCreateCommandQueueWithProperties(ctx, device_id, NULL, &err);
  check_error(err, "creating command queue");

  // TODO load kernel source and build the program

  // TODO create kernel from program

  // TODO create I/O buffers in device memory

  // TODO write input matrices to input buffers

  // TODO set up kernel args

  // TODO execute kernel, read and display results with timing

  // TODO verify results

  // cleanup
  clReleaseCommandQueue(q_cmd);
  clReleaseContext(ctx);

  free(h_out);
  free(h_b);
  free(h_a);

  return 0;
}
