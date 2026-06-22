#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>

static void print_device_info() {
  int deviceCount = 0;
  cudaError_t error = cudaGetDeviceCount(&deviceCount);

  if (error != cudaSuccess) {
    std::cout << "CUDA error: " << cudaGetErrorString(error) << "\n";
    return;
  }

  std::cout << "Number of CUDA devices: " << deviceCount << "\n\n";

  for (int i = 0; i < deviceCount; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    std::cout << "Device " << i << ": " << prop.name << "\n";
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor
              << "\n";
    std::cout << "  Total global memory: " << (prop.totalGlobalMem >> 20)
              << " MB\n\n";
  }
}

void initArray(float* A, int length) {
  for (int i = 0; i < length; i++) {
    A[i] = static_cast<float>(rand()) / RAND_MAX;
  }
}

void serialVecAdd(float* A, float* B, float* C, int length) {
  for (int i = 0; i < length; i++) {
    C[i] = A[i] + B[i];
  }
}

bool vectorApproximatelyEqual(float* A,
                              float* B,
                              int length,
                              float epsilon = 0.001f) {  // 适当增加 epsilon
  for (int i = 0; i < length; i++) {
    float diff = fabsf(A[i] - B[i]);  // 使用 fabsf 针对 float
    if (diff > epsilon) {
      printf("Index %d mismatch: %f != %f (diff=%f)\n", i, A[i], B[i], diff);
      return false;
    }
  }
  return true;
}

__global__ void vecAdd(float* A, float* B, float* C, int vectorLength) {
  int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
  if (workIndex < vectorLength) {
    C[workIndex] = A[workIndex] + B[workIndex];
  }
}

void unifiedMemExample(const int vectorLength) {
  const int threads = 256;
  const int blocks = (vectorLength + threads - 1) / threads;

  float* A = nullptr;
  float* B = nullptr;
  float* C = nullptr;
  float* comparisonResult =
      static_cast<float*>(malloc(vectorLength * sizeof(float)));

  if (cudaMallocManaged(&A, vectorLength * sizeof(float)) != cudaSuccess) {
    printf("cudaMallocManaged failed for A\n");
    return;
  }
  if (cudaMallocManaged(&B, vectorLength * sizeof(float)) != cudaSuccess) {
    printf("cudaMallocManaged failed for B\n");
    return;
  }
  if (cudaMallocManaged(&C, vectorLength * sizeof(float)) != cudaSuccess) {
    printf("cudaMallocManaged failed for C\n");
    return;
  }

  initArray(A, vectorLength);
  initArray(B, vectorLength);

  vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);

  cudaError_t launchErr = cudaGetLastError();
  if (launchErr != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(launchErr));
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);
    return;
  }

  cudaError_t syncErr = cudaDeviceSynchronize();
  if (syncErr != cudaSuccess) {
    printf("Kernel execution failed: %s\n", cudaGetErrorString(syncErr));
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);
    return;
  }

  serialVecAdd(A, B, comparisonResult, vectorLength);

  if (vectorApproximatelyEqual(C, comparisonResult, vectorLength)) {
    printf("Unified Memory Test PASSED: CPU and GPU answers match.\n");
  } else {
    printf("Unified Memory Test FAILED: CPU and GPU answers do not match.\n");
  }

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  free(comparisonResult);
}

int main() {
  srand(static_cast<unsigned int>(time(NULL)));

  const int vectorLength = 1024;

  unifiedMemExample(vectorLength);
  return 0;
}