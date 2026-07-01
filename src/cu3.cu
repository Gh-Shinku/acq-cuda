#include "matrix.hpp"
#include <cuda_runtime.h>
#include <iostream>

__global__ void sgemm_navie_kernel(int num_rows_a, int num_cols_a,
                                   int num_cols_b, float alpha,
                                   const float *matrix_a, const float *matrix_b,
                                   float beta, float *output_matrix) {
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int target_matrix_idx = x * num_cols_a + y;
  int sum = 0;
  for (int k = 0; k < num_cols_a; ++k) {
    sum += matrix_a[x * num_cols_a + k] * matrix_b[k * num_cols_b + y];
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
      int sum = 0;
      for (int k = 0; k < num_cols_a; ++k) {
        sum += matrix_a[i * num_cols_a + k] * matrix_b[j + k * num_cols_b];
      }
      output_matrix[out_idx] = alpha * sum + beta * output_matrix[out_idx];
    }
  }
}

int main() {
  const int num_rows_a = 2;
  const int num_cols_a = 2;
  const int num_cols_b = 3;
  const float alpha = 1;
  const float beta = 0;

  float *matrix_a = new float[num_rows_a * num_cols_a]{1, 2, 3, 4};
  float *matrix_b = new float[num_cols_a * num_cols_b]{1, 2, 3, 4, 5, 6};
  float *matrix_c = new float[num_rows_a * num_cols_b]{};

  Matrix<int> ma(2, 2);
  Matrix<int> mb(2, 3);

  cpu_navie_gemm(num_rows_a, num_cols_a, num_cols_b, alpha, matrix_a, matrix_b,
                 beta, matrix_c);

  for (int i = 0; i < num_rows_a; ++i) {
    for (int j = 0; j < num_cols_b; ++j) {
      std::cout << matrix_c[i * num_cols_a + j] << " ";
    }
  }
  std::cout << "\n";

  delete[] matrix_a;
  delete[] matrix_b;
}