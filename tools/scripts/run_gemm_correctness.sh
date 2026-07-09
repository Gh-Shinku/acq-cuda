#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${repo_dir}/build"
mode="quick"
device="0"
cublas_math="fp32"
cudacxx="/home/zhaoyutong/miniconda3/envs/cuda_ws/bin/nvcc"

usage() {
  cat <<USAGE
Usage: $0 [options]

Options:
  --build-dir PATH  CMake build directory (default: ${repo_dir}/build)
  --device ID       CUDA device id (default: 0)
  --quick           Run quick correctness cases (default)
  --full            Run full correctness cases
  --cublas-math M   cuBLAS math mode: fp32 or default (default: fp32)
  --cudacxx PATH    nvcc path for first-time CMake configure
  --help            Show this help
USAGE
}

is_nonnegative_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

require_value() {
  local name=$1
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "${name} requires a value" >&2
    exit 2
  fi
  printf '%s' "$2"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      build_dir="$(require_value "$1" "${2:-}")"
      shift 2
      ;;
    --device)
      device="$(require_value "$1" "${2:-}")"
      if ! is_nonnegative_int "${device}"; then
        echo "--device must be a non-negative integer, got '${device}'" >&2
        exit 2
      fi
      shift 2
      ;;
    --quick)
      mode="quick"
      shift
      ;;
    --full)
      mode="full"
      shift
      ;;
    --cublas-math)
      cublas_math="$(require_value "$1" "${2:-}")"
      if [[ "${cublas_math}" != "fp32" && "${cublas_math}" != "default" ]]; then
        echo "--cublas-math must be 'fp32' or 'default', got '${cublas_math}'" >&2
        exit 2
      fi
      shift 2
      ;;
    --cudacxx)
      cudacxx="$(require_value "$1" "${2:-}")"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "run '$0 --help' for usage" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
  cmake -S "${repo_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="${cudacxx}"
fi

cmake --build "${build_dir}" --target gemm_correctness_test

"${build_dir}/gemm_correctness_test" \
  "--${mode}" \
  --cublas-math "${cublas_math}" \
  --device "${device}"
