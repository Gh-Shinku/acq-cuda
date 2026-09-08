cmake_minimum_required(VERSION 3.22)

project(CudaDeviceQuery LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
    "Build type for single-config generators" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

option(ACQ_CUDA_LINE_INFO
  "Emit CUDA source line information for Nsight Compute without disabling optimization"
  ON)

add_subdirectory(third_party/Catch2)
enable_testing()
add_executable(matrix_test tests/matrix_test.cpp)
target_link_libraries(matrix_test PRIVATE Catch2::Catch2WithMain)

add_executable(cu1 src/misc/cu1.cu)
add_executable(cu2 src/misc/cu2.cu)
add_executable(cu3 src/misc/cu3.cu)
add_executable(gemm_benchmark benchmarks/gemm_benchmark.cu)
add_executable(hgemm_benchmark benchmarks/hgemm_benchmark.cu)
add_executable(gemm_profile tools/profile/gemm_profile.cu)
add_executable(gemm_correctness_test tests/gemm_correctness_test.cu)
add_executable(device_buffer_test tests/device_buffer_test.cu)
add_executable(tanh_correctness_test tests/tanh_correctness_test.cu)
add_library(acq_cuda_ops STATIC src/tanh.cu)
add_library(gemm_kernels STATIC
  src/gemm/registry.cu
  src/gemm/sgemm_async_tiling.cu
  src/gemm/sgemm_cublas.cu
  src/gemm/sgemm_cutlass.cu
  src/gemm/sgemm_naive.cu
  src/gemm/sgemm_smem.cu
  src/gemm/sgemm_thread_tiling.cu
  src/gemm/sgemm_wmma_tf32.cu
  src/gemm/sgemm_warp_tiling.cu
)
add_library(hgemm_fp16_4096_kernels STATIC
  src/gemm/hgemm_fp16_4096_cublas.cu
  src/gemm/hgemm_fp16_4096_custom.cu
  src/gemm/hgemm_fp16_4096_registry.cu
)

add_library(acq_project_headers INTERFACE)
target_include_directories(acq_project_headers INTERFACE
  "${CMAKE_SOURCE_DIR}/include"
)

add_library(cutlass_headers INTERFACE)
target_include_directories(cutlass_headers INTERFACE
  "${CMAKE_SOURCE_DIR}/third_party/cutlass/include"
  "${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include"
)

foreach(target matrix_test cu1 cu2 cu3 gemm_benchmark gemm_profile
  hgemm_benchmark gemm_correctness_test device_buffer_test tanh_correctness_test
  gemm_kernels hgemm_fp16_4096_kernels acq_cuda_ops)
  target_link_libraries(${target} PRIVATE acq_project_headers)
endforeach()

foreach(target gemm_kernels cut1 cut2)
  if(TARGET ${target})
    target_link_libraries(${target} PRIVATE cutlass_headers)
    target_compile_options(${target} PRIVATE --expt-relaxed-constexpr)
  endif()
endforeach()

if(ACQ_CUDA_LINE_INFO)
  target_compile_options(gemm_kernels PRIVATE --generate-line-info)
  target_compile_options(hgemm_fp16_4096_kernels PRIVATE --generate-line-info)
endif()

target_link_libraries(gemm_kernels PUBLIC CUDA::cublas)
target_link_libraries(hgemm_fp16_4096_kernels PUBLIC CUDA::cublas)
target_link_libraries(acq_cuda_ops PUBLIC CUDA::cudart)

# This target intentionally targets the fixed RTX 5090/Blackwell workload.
set_target_properties(hgemm_fp16_4096_kernels PROPERTIES CUDA_ARCHITECTURES 120)

target_link_libraries(cu3 PRIVATE gemm_kernels)
target_link_libraries(gemm_benchmark PRIVATE gemm_kernels)
target_link_libraries(hgemm_benchmark PRIVATE hgemm_fp16_4096_kernels)
target_link_libraries(gemm_profile PRIVATE gemm_kernels)
target_link_libraries(gemm_correctness_test PRIVATE gemm_kernels)
target_link_libraries(device_buffer_test PRIVATE Catch2::Catch2WithMain CUDA::cudart)
target_link_libraries(tanh_correctness_test PRIVATE Catch2::Catch2WithMain acq_cuda_ops)

add_test(NAME gemm_correctness
  COMMAND gemm_correctness_test --quick)
add_test(NAME device_buffer
  COMMAND device_buffer_test)
add_test(NAME tanh_correctness
  COMMAND tanh_correctness_test)

set(CLANGD_COMPILE_COMMANDS_JSON
  "${CMAKE_BINARY_DIR}/compile_commands.clangd.json")
set(CLANGD_COMPILE_COMMANDS_DIR "${CMAKE_BINARY_DIR}/clangd")
set(CLANGD_COMPILE_COMMANDS_LINK
  "${CLANGD_COMPILE_COMMANDS_DIR}/compile_commands.json")

add_custom_command(
  OUTPUT "${CLANGD_COMPILE_COMMANDS_JSON}"
  BYPRODUCTS "${CLANGD_COMPILE_COMMANDS_LINK}"
  COMMAND
  "${CMAKE_SOURCE_DIR}/tools/scripts/filter-compile-commands-clangd.sh"
  "${CMAKE_BINARY_DIR}/compile_commands.json"
  "${CLANGD_COMPILE_COMMANDS_JSON}"
  "${CLANGD_COMPILE_COMMANDS_DIR}"
  "${CMAKE_SOURCE_DIR}"
  "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
  DEPENDS
  "${CMAKE_BINARY_DIR}/compile_commands.json"
  "${CMAKE_SOURCE_DIR}/tools/scripts/filter-compile-commands-clangd.sh"
  COMMENT "Generating clangd compile_commands.json"
  VERBATIM
)

add_custom_target(clangd_compile_commands ALL
  DEPENDS "${CLANGD_COMPILE_COMMANDS_JSON}")
