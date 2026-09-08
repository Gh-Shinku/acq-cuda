#include "gemm/hgemm_fp16_4096.hpp"

namespace gemm::fp16_4096 {

const std::vector<Implementation>& implementations() {
  static const std::vector<Implementation> values = {
      {"cuBLAS GemmEx FP16", launch_cublas, true},
      {"Custom FP16", launch_custom, false},
  };
  return values;
}

}  // namespace gemm::fp16_4096
