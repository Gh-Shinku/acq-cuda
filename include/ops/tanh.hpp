#ifndef OPS_TANH_HPP
#define OPS_TANH_HPP

#include <cuda_runtime_api.h>

#include <cstddef>

namespace ops {

void launch_tanh(const float* x, float* y, std::size_t n,
                 cudaStream_t stream = nullptr);

}  // namespace ops

#endif  // OPS_TANH_HPP
