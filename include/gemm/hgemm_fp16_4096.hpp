#ifndef GEMM_HGEMM_FP16_4096_HPP
#define GEMM_HGEMM_FP16_4096_HPP

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <vector>

namespace gemm::fp16_4096 {

inline constexpr int kDimension = 4096;

// The benchmark contract is row-major, contiguous D = A * B, with all three
// matrices kDimension by kDimension. A and B are FP16; multiplication
// accumulates in FP32 and D is rounded to FP16.
using Launcher = void (*)(const __half* a, const __half* b, __half* d,
                          cudaStream_t stream);

struct Implementation {
  const char* name;
  Launcher launcher;
  bool is_baseline;
};

void launch_cublas(const __half* a, const __half* b, __half* d,
                   cudaStream_t stream);

// Replace the implementation behind this stable entry point while tuning a
// hand-written kernel. Keep the input/output layout and stream semantics.
void launch_custom(const __half* a, const __half* b, __half* d,
                   cudaStream_t stream);

const std::vector<Implementation>& implementations();

}  // namespace gemm::fp16_4096

#endif  // GEMM_HGEMM_FP16_4096_HPP
