#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${repo_dir}/build"
impl="CUDA Tiling"
size="1024"
m=""
n=""
k=""
warmup="10"
repeat="1"
device="0"
ncu_bin="ncu"
ncu_set="full"
output=""
cudacxx="/home/zhaoyutong/miniconda3/envs/cuda_ws/bin/nvcc"

usage() {
  cat <<USAGE
Usage: $0 [options]

Options:
  --impl NAME        GEMM implementation name (default: CUDA Tiling)
  --size N           Square GEMM size (default: 1024)
  --m M              GEMM M dimension; must be used with --n and --k
  --n N              GEMM N dimension; must be used with --m and --k
  --k K              GEMM K dimension; must be used with --m and --n
  --warmup N         Warmup iterations outside ncu range (default: 10)
  --repeat N         Profiled launches inside ncu range (default: 1)
  --device ID        CUDA device id (default: 0)
  --ncu PATH         ncu executable (default: ncu)
  --ncu-set NAME     ncu section set (default: full)
  --output PATH      ncu output path without .ncu-rep suffix
  --build-dir PATH   CMake build directory (default: ${repo_dir}/build)
  --cudacxx PATH     nvcc path for first-time CMake configure
  --help             Show this help
USAGE
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
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
    --impl)
      impl="$(require_value "$1" "${2:-}")"
      shift 2
      ;;
    --size)
      size="$(require_value "$1" "${2:-}")"
      if ! is_positive_int "${size}"; then
        echo "--size must be a positive integer, got '${size}'" >&2
        exit 2
      fi
      shift 2
      ;;
    --m)
      m="$(require_value "$1" "${2:-}")"
      if ! is_positive_int "${m}"; then
        echo "--m must be a positive integer, got '${m}'" >&2
        exit 2
      fi
      shift 2
      ;;
    --n)
      n="$(require_value "$1" "${2:-}")"
      if ! is_positive_int "${n}"; then
        echo "--n must be a positive integer, got '${n}'" >&2
        exit 2
      fi
      shift 2
      ;;
    --k)
      k="$(require_value "$1" "${2:-}")"
      if ! is_positive_int "${k}"; then
        echo "--k must be a positive integer, got '${k}'" >&2
        exit 2
      fi
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
    --ncu)
      ncu_bin="$(require_value "$1" "${2:-}")"
      shift 2
      ;;
    --ncu-set)
      ncu_set="$(require_value "$1" "${2:-}")"
      shift 2
      ;;
    --output)
      output="$(require_value "$1" "${2:-}")"
      shift 2
      ;;
    --build-dir)
      build_dir="$(require_value "$1" "${2:-}")"
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

impl_slug="$(printf '%s' "${impl}" | tr '[:upper:] ' '[:lower:]_')"
if [[ -n "${m}${n}${k}" ]]; then
  if [[ -z "${m}" || -z "${n}" || -z "${k}" ]]; then
    echo "--m, --n, and --k must be set together" >&2
    exit 1
  fi
  shape_slug="${m}x${n}x${k}"
  profile_args=(--m "${m}" --n "${n}" --k "${k}")
else
  shape_slug="${size}"
  profile_args=(--size "${size}")
fi

if [[ -z "${output}" ]]; then
  output="${repo_dir}/ncu_reports/${impl_slug}_${shape_slug}"
fi
mkdir -p "$(dirname "${output}")"

if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
  cmake -S "${repo_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="${cudacxx}"
fi

cmake --build "${build_dir}" --target gemm_profile

"${ncu_bin}" \
  --profile-from-start off \
  --set "${ncu_set}" \
  -o "${output}" \
  "${build_dir}/gemm_profile" \
  --impl "${impl}" \
  "${profile_args[@]}" \
  --warmup "${warmup}" \
  --repeat "${repeat}" \
  --device "${device}"
