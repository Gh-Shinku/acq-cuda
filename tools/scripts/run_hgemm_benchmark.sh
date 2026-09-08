#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${repo_dir}/build"
cudacxx="/home/zhaoyutong/miniconda3/envs/cuda_ws/bin/nvcc"

usage() {
  cat <<USAGE
Usage: $0 [--build-dir PATH] [--cudacxx PATH] [hgemm_benchmark options]

Builds the fixed RTX 5090 FP16 GEMM benchmark, then forwards all remaining
options to hgemm_benchmark. For benchmark options run:
  $0 --help
USAGE
}

benchmark_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      [[ $# -ge 2 ]] || { echo "--build-dir requires a value" >&2; exit 2; }
      build_dir="$2"
      shift 2
      ;;
    --cudacxx)
      [[ $# -ge 2 ]] || { echo "--cudacxx requires a value" >&2; exit 2; }
      cudacxx="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      benchmark_args+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
  cmake -S "${repo_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="${cudacxx}"
fi

cmake --build "${build_dir}" --target hgemm_benchmark
exec "${build_dir}/hgemm_benchmark" "${benchmark_args[@]}"
