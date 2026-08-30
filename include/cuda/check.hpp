#ifndef ACQ_CUDA_CHECK_HPP
#define ACQ_CUDA_CHECK_HPP

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>

namespace acq::cuda {

inline void check(cudaError_t status, const char* expression, const char* file,
                  int line) {
  if (status == cudaSuccess) {
    return;
  }

  std::ostringstream message;
  message << "CUDA error at " << file << ":" << line << " while running "
          << expression << ": " << cudaGetErrorString(status);
  throw std::runtime_error(message.str());
}

}  // namespace acq::cuda

#define ACQ_CUDA_CHECK(expression) \
  ::acq::cuda::check((expression), #expression, __FILE__, __LINE__)

#endif  // ACQ_CUDA_CHECK_HPP
