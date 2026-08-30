#include "ops/tanh.hpp"

#include <cmath>

namespace ops {
namespace {

__global__ void tanh_kernel(const float* x, float* y, std::size_t n) {
  std::size_t const idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;

  if (idx < n) {
    y[idx] = tanhf(x[idx]);
  }
}

}  // namespace

void launch_tanh(const float* x, float* y, std::size_t n, cudaStream_t stream) {
  if (n == 0) {
    return;
  }

  constexpr unsigned int kThreadsPerBlock = 256;
  dim3 const block_dim{kThreadsPerBlock};
  dim3 const grid_dim{
      static_cast<unsigned int>((n + kThreadsPerBlock - 1) / kThreadsPerBlock)};

  tanh_kernel<<<grid_dim, block_dim, 0, stream>>>(x, y, n);
}

}  // namespace ops
