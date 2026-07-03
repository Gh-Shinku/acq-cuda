#include "gemm/check.hpp"
#include "gemm/sgemm.hpp"
#include "matrix.hpp"
#include "utils.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

void cpu_naive_gemm(int num_rows_a, int num_cols_a, int num_cols_b, float alpha,
                    const float* matrix_a, const float* matrix_b,
                    const float* matrix_c, float beta, float* output_matrix) {
  for (int i = 0; i < num_rows_a; ++i) {
    for (int j = 0; j < num_cols_b; ++j) {
      const int out_idx = i * num_cols_b + j;
      float sum = 0.0f;
      for (int k = 0; k < num_cols_a; ++k) {
        sum += matrix_a[i * num_cols_a + k] * matrix_b[k * num_cols_b + j];
      }
      output_matrix[out_idx] = alpha * sum + beta * matrix_c[out_idx];
    }
  }
}

template <Numeric T>
class CudaMemory {
 public:
  explicit CudaMemory(size_t len) : len_(len), size_(sizeof(T) * len) {
    cudaMalloc(&mem_, size_);
  }

  ~CudaMemory() { cudaFree(mem_); }

  CudaMemory(const CudaMemory&) = delete;
  CudaMemory& operator=(const CudaMemory&) = delete;

  T* ptr() const { return mem_; }
  size_t getSize() const { return size_; }

 private:
  T* mem_ = nullptr;
  size_t len_ = 0;
  size_t size_ = 0;
};

bool matrix_near(const Matrix<float>& lhs, const Matrix<float>& rhs,
                 float atol = 1.0e-4f, float rtol = 1.0e-4f) {
  if (lhs.row() != rhs.row() || lhs.col() != rhs.col()) {
    return false;
  }

  for (int i = 0; i < lhs.row(); ++i) {
    for (int j = 0; j < lhs.col(); ++j) {
      float expected = lhs(i, j);
      float actual = rhs(i, j);
      float tolerance = atol + rtol * std::abs(expected);
      if (std::abs(expected - actual) > tolerance) {
        return false;
      }
    }
  }
  return true;
}

int main() {
  try {
    const int num_rows_a = 2;
    const int num_cols_a = 2;
    const int num_cols_b = 3;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    RandomGenerator<float> rg(1, 10);
    Matrix<float> cpu_a(num_rows_a, num_cols_a, rg);
    Matrix<float> cpu_b(num_cols_a, num_cols_b, rg);
    Matrix<float> cpu_c(num_rows_a, num_cols_b, 0.0f);
    Matrix<float> cpu_out(num_rows_a, num_cols_b, 0.0f);
    Matrix<float> cuda_out(num_rows_a, num_cols_b, 0.0f);

    CudaMemory<float> cuda_a(num_rows_a * num_cols_a);
    CudaMemory<float> cuda_b(num_cols_a * num_cols_b);
    CudaMemory<float> cuda_c(num_rows_a * num_cols_b);
    CudaMemory<float> cuda_d(num_rows_a * num_cols_b);

    GEMM_CUDA_CHECK(cudaMemcpy(cuda_a.ptr(), cpu_a.ptr(), cuda_a.getSize(),
                               cudaMemcpyHostToDevice));
    GEMM_CUDA_CHECK(cudaMemcpy(cuda_b.ptr(), cpu_b.ptr(), cuda_b.getSize(),
                               cudaMemcpyHostToDevice));
    GEMM_CUDA_CHECK(cudaMemcpy(cuda_c.ptr(), cpu_c.ptr(), cuda_c.getSize(),
                               cudaMemcpyHostToDevice));
    GEMM_CUDA_CHECK(cudaMemset(cuda_d.ptr(), 0, cuda_d.getSize()));

    Matrix<float> matrix_out = cpu_a * cpu_b;
    cpu_naive_gemm(num_rows_a, num_cols_a, num_cols_b, alpha, cpu_a.ptr(),
                   cpu_b.ptr(), cpu_c.ptr(), beta, cpu_out.ptr());

    gemm::SgemmProblem problem{num_rows_a, num_cols_b, num_cols_a, alpha, beta};
    gemm::launch_sgemm_naive(problem, cuda_a.ptr(), cuda_b.ptr(), cuda_c.ptr(),
                             cuda_d.ptr(), nullptr);
    GEMM_CUDA_CHECK(cudaMemcpy(cuda_out.ptr(), cuda_d.ptr(), cuda_d.getSize(),
                               cudaMemcpyDeviceToHost));

    std::cout << "[cpu_naive_gemm]"
              << (matrix_near(cpu_out, matrix_out) ? "EQ" : "NEQ") << "\n";
    std::cout << "[cuda_naive_gemm]"
              << (matrix_near(cuda_out, cpu_out) ? "EQ" : "NEQ") << "\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "cu3: " << e.what() << "\n";
    return 1;
  }
}
