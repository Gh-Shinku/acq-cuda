#include "gemm/check.hpp"
#include "gemm/sgemm.hpp"

#include <cuda_runtime.h>

namespace gemm {

constexpr int block_size = 16;
constexpr int tile_size = block_size;

namespace {

__global__ void sgemm_smem_kernel(SgemmProblem problem, const float *a,
                                  const float *b, const float *c, float *d) {
  __shared__ float tile_a[tile_size * tile_size];
  __shared__ float tile_b[tile_size * tile_size];

  /* thread index in grid */
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  float sum = 0;
  a += blockIdx.y * blockDim.y * problem.k;
  b += blockIdx.x * blockDim.x;

  /* iterate `ceil(K / tile_size)` rounds */
  for (int tile_idx = 0; tile_idx < problem.k; tile_idx += tile_size) {
    /* copy data from matrix to tile */
    if (row < problem.m && tile_idx + threadIdx.x < problem.k) {
      tile_a[threadIdx.y * tile_size + threadIdx.x] =
          a[tile_idx + threadIdx.y * problem.k + threadIdx.x];
    } else {
      tile_a[threadIdx.y * tile_size + threadIdx.x] = 0;
    }
    if (tile_idx + threadIdx.y < problem.k && col < problem.n) {
      tile_b[threadIdx.y * tile_size + threadIdx.x] =
          b[(tile_idx + threadIdx.y) * problem.n + threadIdx.x];
    } else {
      tile_b[threadIdx.y * tile_size + threadIdx.x] = 0;
    }
    /* wait for tile data to be prepared */
    __syncthreads();

    /* dot product */
    for (int dot_idx = 0; dot_idx < tile_size; ++dot_idx) {
      sum += tile_a[threadIdx.y * tile_size + dot_idx] *
             tile_b[dot_idx * tile_size + threadIdx.x];
    }

    /* wait to avoid fetching next tile */
    __syncthreads();
  }

  if (row < problem.m && col < problem.n) {
    d[row * problem.n + col] =
        problem.alpha * sum + problem.beta * c[row * problem.n + col];
  }
}
} // namespace

void launch_sgemm_smem(const SgemmProblem &problem, const float *a,
                       const float *b, const float *c, float *d,
                       cudaStream_t stream) {
  dim3 block(block_size, block_size);
  dim3 grid((problem.n + block.x - 1) / block.x,
            (problem.m + block.y - 1) / block.y);

  sgemm_smem_kernel<<<grid, block, 0, stream>>>(problem, a, b, c, d);
  GEMM_CUDA_CHECK(cudaGetLastError());
}

} // namespace gemm
