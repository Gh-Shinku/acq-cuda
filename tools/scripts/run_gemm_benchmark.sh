#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${repo_dir}/build"
results_dir="${repo_dir}/benchmark_results"
csv_path=""
sizes="64,96,128,256,512,768,1024,1536,2048,3072,4096"
warmup="10"
repeat=""
device="0"
python_command="/home/zhaoyutong/miniconda3/bin/conda run -n base python"
cudacxx="/home/zhaoyutong/miniconda3/envs/cuda_ws/bin/nvcc"
skip_correctness="false"

usage() {
  cat <<USAGE
Usage: $0 [options]

Options:
  --build-dir PATH       CMake build directory (default: ${repo_dir}/build)
  --results-dir PATH     Benchmark results directory (default: ${repo_dir}/benchmark_results)
  --csv PATH             CSV output path (default: <results-dir>/gemm_benchmark_results.csv)
  --sizes LIST           Comma-separated square sizes
  --warmup N             Warmup iterations (default: 10)
  --repeat N             Repeat iterations; omit to use benchmark defaults
  --device ID            CUDA device id (default: 0)
  --python COMMAND       Python command for plotting
  --cudacxx PATH         nvcc path for first-time CMake configure
  --skip-correctness     Run benchmark without the quick correctness gate
  --help                 Show this help
USAGE
}

is_nonnegative_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
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
    --results-dir)
      results_dir="$(require_value "$1" "${2:-}")"
      shift 2
      ;;
    --csv)
      csv_path="$(require_value "$1" "${2:-}")"
      shift 2
      ;;
    --sizes)
      sizes="$(require_value "$1" "${2:-}")"
      shift 2
      ;;
    --warmup)
      warmup="$(require_value "$1" "${2:-}")"
      if ! is_nonnegative_int "${warmup}"; then
        echo "--warmup must be a non-negative integer, got '${warmup}'" >&2
        exit 2
      fi
      shift 2
      ;;
    --repeat)
      repeat="$(require_value "$1" "${2:-}")"
      if ! is_positive_int "${repeat}"; then
        echo "--repeat must be a positive integer, got '${repeat}'" >&2
        exit 2
      fi
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
    --python)
      python_command="$(require_value "$1" "${2:-}")"
      shift 2
      ;;
    --cudacxx)
      cudacxx="$(require_value "$1" "${2:-}")"
      shift 2
      ;;
    --skip-correctness)
      skip_correctness="true"
      shift
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

if [[ -z "${csv_path}" ]]; then
  csv_path="${results_dir}/gemm_benchmark_results.csv"
fi

IFS=' ' read -r -a python_cmd <<< "${python_command}"

mkdir -p "${results_dir}"

if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
  cmake -S "${repo_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="${cudacxx}"
fi

if [[ "${skip_correctness}" == "false" ]]; then
  "${repo_dir}/tools/scripts/run_gemm_correctness.sh" \
    --build-dir "${build_dir}" \
    --device "${device}" \
    --quick \
    --cudacxx "${cudacxx}"
fi

cmake --build "${build_dir}" --target gemm_benchmark

benchmark_args=(
  --sizes "${sizes}"
  --warmup "${warmup}"
  --device "${device}"
  --csv "${csv_path}"
)

if [[ -n "${repeat}" ]]; then
  benchmark_args+=(--repeat "${repeat}")
fi

"${build_dir}/gemm_benchmark" "${benchmark_args[@]}"

"${python_cmd[@]}" "${repo_dir}/tools/scripts/plot_gemm_benchmark.py" \
  --csv "${csv_path}" \
  --output "${results_dir}/gemm_benchmark_summary.png"
