#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build}"
results_dir="${RESULTS_DIR:-${repo_dir}/benchmark_results}"
csv_path="${CSV_PATH:-${results_dir}/gemm_benchmark_results.csv}"
sizes="${SIZES:-64,96,128,256,512,768,1024,1536,2048,3072,4096}"
warmup="${WARMUP:-10}"
repeat="${REPEAT:-}"
device="${DEVICE:-0}"
python_bin="${PYTHON:-/home/zhaoyutong/miniconda3/bin/conda run -n base python}"

mkdir -p "${results_dir}"

if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
  CUDACXX="${CUDACXX:-/home/zhaoyutong/miniconda3/envs/cuda_ws/bin/nvcc}" \
    cmake -S "${repo_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
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

${python_bin} "${repo_dir}/scripts/plot_gemm_benchmark.py" \
  --csv "${csv_path}" \
  --output "${results_dir}/gemm_benchmark_summary.png"
