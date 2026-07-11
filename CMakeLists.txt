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

add_executable(cu1 src/cu1.cu)
add_executable(cu2 src/cu2.cu)
add_executable(cu3 src/cu3.cu)
add_executable(gemm_benchmark benchmarks/gemm_benchmark.cu)
add_executable(gemm_profile tools/profile/gemm_profile.cu)
add_executable(gemm_correctness_test tests/gemm_correctness_test.cu)
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

foreach(target matrix_test cu1 cu2 cu3 gemm_benchmark gemm_profile
  gemm_correctness_test)
  target_include_directories(${target} PRIVATE "${CMAKE_SOURCE_DIR}/include")
endforeach()

target_include_directories(gemm_kernels PUBLIC
  "${CMAKE_SOURCE_DIR}/include")
target_include_directories(gemm_kernels PRIVATE
  "${CMAKE_SOURCE_DIR}/third_party/cutlass/include"
  "${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include"
)
target_compile_options(gemm_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-O2>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${ACQ_CUDA_LINE_INFO}>>:--generate-line-info>)
target_link_libraries(gemm_kernels PUBLIC CUDA::cublas)

target_link_libraries(cu3 PRIVATE gemm_kernels)
target_link_libraries(gemm_benchmark PRIVATE gemm_kernels)
target_link_libraries(gemm_profile PRIVATE gemm_kernels)
target_link_libraries(gemm_correctness_test PRIVATE gemm_kernels)

add_test(NAME gemm_correctness
  COMMAND gemm_correctness_test --quick)

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
