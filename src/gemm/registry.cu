#include "gemm/sgemm.hpp"

namespace gemm {

const std::vector<SgemmImplementation> &get_sgemm_implementations() {
  static const std::vector<SgemmImplementation> implementations = {
      {"cuBLAS", launch_sgemm_cublas, true, SgemmAccuracy::kFp32},
      {"CUTLASS", launch_sgemm_cutlass, false, SgemmAccuracy::kFp32},
      {"CUTLASS TensorOp TF32", launch_sgemm_cutlass_tensorop_tf32, false,
       SgemmAccuracy::kTf32Approx},
      {"CUDA Naive", launch_sgemm_naive, false, SgemmAccuracy::kFp32},
      {"CUDA SMEM", launch_sgemm_smem, false, SgemmAccuracy::kFp32},
      {"CUDA Thread Tiling", launch_sgemm_thread_tiling, false,
       SgemmAccuracy::kFp32},
      {"CUDA Warp Tiling", launch_sgemm_warp_tiling, false,
       SgemmAccuracy::kFp32},
      {"CUDA Async Tiling", launch_sgemm_async_tiling, false,
       SgemmAccuracy::kFp32},
      {"CUDA WMMA TF32", launch_sgemm_wmma_tf32, false,
       SgemmAccuracy::kTf32Approx}};
  return implementations;
}

} // namespace gemm
