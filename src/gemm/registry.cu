#include "gemm/sgemm.hpp"

namespace gemm {

const std::vector<SgemmImplementation>& get_sgemm_implementations() {
  static const std::vector<SgemmImplementation> implementations = {
      {"CUTLASS", launch_sgemm_cutlass, true},
      {"CUDA Naive", launch_sgemm_naive, false},
  };
  return implementations;
}

}  // namespace gemm
