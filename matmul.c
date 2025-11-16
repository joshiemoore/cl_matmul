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
  cl_int err;
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

  // load kernel source
  FILE* inf = fopen("matmul.cl", "rb");
  fseek(inf, 0, SEEK_END);
  long inf_size = ftell(inf);
  fseek(inf, 0, SEEK_SET);
  char* kernel_src = malloc(inf_size + 1);
  fread(kernel_src, inf_size, 1, inf);
  kernel_src[inf_size] = 0;
  fclose(inf);

  // build the program
  program = clCreateProgramWithSource(ctx, 1, (const char**)&kernel_src, NULL, &err);
  check_error(err, "creating program");
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t err_len;
    char err_buf[4096];
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(err_buf), err_buf, &err_len);
    printf("ERROR BUILDING PROGRAM:\n\n%s", err_buf);
    return 1;
  }

  // create kernel from program
  ko_matmul = clCreateKernel(program, "matmul", &err);
  check_error(err, "creating kernel");

  // TODO create I/O buffers in device memory

  // TODO write input matrices to input buffers

  // TODO set up kernel args

  // TODO execute kernel, read and display results with timing

  // TODO verify results

  // cleanup
  clReleaseKernel(ko_matmul);
  clReleaseProgram(program);
  clReleaseCommandQueue(q_cmd);
  clReleaseContext(ctx);

  free(kernel_src);
  free(h_out);
  free(h_b);
  free(h_a);

  return 0;
}
