#include <cuda_runtime.h>
#include <iostream>
#include <random>

__global__ void cuda_vecAdd(float *A, float *B, float *C, int vectorLength) {
  int workIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (workIndex < vectorLength) {
    C[workIndex] = A[workIndex] + B[workIndex];
  }
}

void cpu_vecAdd(float *A, float *B, float *C, int vectorLength) {
  for (int i = 0; i < vectorLength; ++i) {
    C[i] = A[i] + B[i];
  }
}

class TestVecAdd {
public:
  TestVecAdd() {
    cpu_A = new float[vector_len];
    cpu_B = new float[vector_len];
    cpu_C1 = new float[vector_len];
    cpu_C2 = new float[vector_len];
    cudaMalloc(&cuda_A, vector_size);
    cudaMalloc(&cuda_B, vector_size);
    cudaMalloc(&cuda_C, vector_size);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < vector_len; ++i) {
      cpu_A[i] = dist(gen);
      cpu_B[i] = dist(gen);
    }

    cudaMemcpy(cuda_A, cpu_A, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B, cpu_B, vector_size, cudaMemcpyHostToDevice);
  }

  ~TestVecAdd() {
    delete[] cpu_A;
    delete[] cpu_B;
    delete[] cpu_C1;
    delete[] cpu_C2;
    cudaFree(cuda_A);
    cudaFree(cuda_B);
    cudaFree(cuda_C);
  }

  void check() {
    cpu_vecAdd(cpu_A, cpu_B, cpu_C1, vector_len);
    cuda_vecAdd<<<grid_dim, block_dim>>>(cuda_A, cuda_B, cuda_C, vector_len);
    cudaMemcpy(cpu_C2, cuda_C, vector_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < vector_len; ++i) {
      if (cpu_C1[i] != cpu_C2[i]) {
        std::cout << "[FAIL] cpu_vecAdd and cuda_vecAdd do not match\n";
        return;
      }
    }
    std::cout << "[SUCC] cpu_vecAdd and cuda_vecAdd match\n";
  }

private:
  static constexpr int grid_dim = 32;
  static constexpr int block_dim = 32;
  static constexpr int vector_len = 1024;
  static constexpr int vector_size = vector_len * sizeof(float);
  float *cuda_A, *cuda_B, *cuda_C;
  float *cpu_A, *cpu_B, *cpu_C1, *cpu_C2;
};

int main() {
  TestVecAdd t;
  t.check();
}