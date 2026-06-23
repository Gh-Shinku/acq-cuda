cmake_minimum_required(VERSION 3.18)

project(CudaDeviceQuery LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(cu1 src/cu1.cu)
add_executable(cu2 src/cu2.cu)

execute_process(
  COMMAND
    "${CMAKE_SOURCE_DIR}/scripts/filter-compile-commands-clangd.sh"
    "${CMAKE_BINARY_DIR}/compile_commands.json"
    "${CMAKE_BINARY_DIR}/compile_commands.clangd.json"
    "${CMAKE_BINARY_DIR}/clangd"
  RESULT_VARIABLE CLANGD_COMPILE_COMMANDS_RESULT
)
if(NOT CLANGD_COMPILE_COMMANDS_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to generate clangd compile_commands.json")
endif()
