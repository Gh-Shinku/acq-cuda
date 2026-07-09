#include "gemm/sgemm.hpp"

namespace gemm {

const std::vector<SgemmImplementation> &get_sgemm_implementations() {
  static const std::vector<SgemmImplementation> implementations = {
      {"cuBLAS", launch_sgemm_cublas, true},
      {"CUTLASS", launch_sgemm_cutlass, false},
      {"CUDA Naive", launch_sgemm_naive, false},
      {"CUDA SMEM", launch_sgemm_smem, false}};
  return implementations;
}

} // namespace gemm
