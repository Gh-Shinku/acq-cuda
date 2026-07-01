#include "matrix.hpp"
#include "utils.hpp"
#include <cuda_runtime.h>
#include <iostream>

__global__ void sgemm_navie_kernel(int num_rows_a, int num_cols_a,
                                   int num_cols_b, float alpha,
                                   const float *matrix_a, const float *matrix_b,
                                   float beta, float *output_matrix) {
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = threadIdx.y + blockIdx.y * blockDim.y;
  const int target_matrix_idx = row * num_cols_b + col;
  if (row >= num_rows_a || col >= num_cols_b) {
    return;
  }
  float sum = 0;
  for (int k = 0; k < num_cols_a; ++k) {
    sum += matrix_a[row * num_cols_a + k] * matrix_b[k * num_cols_b + col];
  }
  output_matrix[target_matrix_idx] =
      alpha * sum + beta * output_matrix[target_matrix_idx];
}

void cpu_navie_gemm(int num_rows_a, int num_cols_a, int num_cols_b, float alpha,
                    const float *matrix_a, const float *matrix_b, float beta,
                    float *output_matrix) {
  for (int i = 0; i < num_rows_a; ++i) {
    for (int j = 0; j < num_cols_b; ++j) {
      const int out_idx = i * num_cols_b + j;
      float sum = 0;
      for (int k = 0; k < num_cols_a; ++k) {
        sum += matrix_a[i * num_cols_a + k] * matrix_b[j + k * num_cols_b];
      }
      output_matrix[out_idx] = alpha * sum + beta * output_matrix[out_idx];
    }
  }
}

template <Numeric T> class CudaMemory {
private:
  T *mem_;
  size_t len_;
  size_t size_;

public:
  CudaMemory(size_t len) : len_(len), size_(sizeof(T) * len) {
    cudaMalloc(&mem_, size_);
  }

  ~CudaMemory() { cudaFree(mem_); }

  T *ptr() const { return mem_; }

  size_t getSize() { return size_; }
};

int main() {
  /* constants */
  const int num_rows_a = 2;
  const int num_cols_a = 2;
  const int num_cols_b = 3;
  const float alpha = 1;
  const float beta = 0;
  dim3 block_dim = {16, 16};
  dim3 grid_dim = {(num_rows_a + block_dim.x - 1) / block_dim.x,
                   (num_cols_b + block_dim.y - 1) / block_dim.y};

  /* allocate cpu memory */
  RandomGenerator<float> rg(1, 10);
  Matrix<float> cpu_a(num_rows_a, num_cols_a, rg);
  Matrix<float> cpu_b(num_cols_a, num_cols_b, rg);
  Matrix<float> cpu_c(num_rows_a, num_cols_b, 0);
  Matrix<float> cuda2cpu_c(num_rows_a, num_cols_b);

  const float *matrix_a = cpu_a.ptr();
  const float *matrix_b = cpu_b.ptr();
  float *matrix_c = cpu_c.ptr();

  /* allocate gpu memory */
  CudaMemory<float> cuda_a(num_rows_a * num_cols_a),
      cuda_b(num_cols_a * num_cols_b), cuda_c(num_rows_a * num_cols_b);
  cudaMemset(cuda_c.ptr(), 0, cuda_c.getSize());
  cudaMemcpy(cuda_a.ptr(), cpu_a.ptr(), cuda_a.getSize(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_b.ptr(), cpu_b.ptr(), cuda_b.getSize(),
             cudaMemcpyHostToDevice);

  /* calculate */
  Matrix<float> cpu_ab = cpu_a * cpu_b;
  cpu_navie_gemm(num_rows_a, num_cols_a, num_cols_b, alpha, matrix_a, matrix_b,
                 beta, matrix_c);
  sgemm_navie_kernel<<<grid_dim, block_dim>>>(num_rows_a, num_cols_a,
                                              num_cols_b, alpha, cuda_a.ptr(),
                                              cuda_b.ptr(), beta, cuda_c.ptr());
  cudaMemcpy(cuda2cpu_c.ptr(), cuda_c.ptr(), cuda_c.getSize(),
             cudaMemcpyDeviceToHost);
  /* test cpu_navie_gemm */
  std::cout << "[cpu_navie_gemm]" << (cpu_c == cpu_ab ? "EQ" : "NEQ") << "\n";
  /* test navie gemm kernel */
  std::cout << "[cuda_navie_gemm]" << (cuda2cpu_c == cpu_c ? "EQ" : "NEQ")
            << "\n";
}