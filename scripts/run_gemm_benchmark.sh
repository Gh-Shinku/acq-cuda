#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build}"
results_dir="${RESULTS_DIR:-${repo_dir}/benchmark_results}"
csv_path="${CSV_PATH:-${results_dir}/gemm_benchmark.csv}"
sizes="${SIZES:-256,512,1024,2048,4096}"
warmup="${WARMUP:-5}"
repeat="${REPEAT:-20}"
device="${DEVICE:-0}"
python_bin="${PYTHON:-/home/zhaoyutong/miniconda3/bin/conda run -n base python}"

mkdir -p "${results_dir}"

if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
  CUDACXX="${CUDACXX:-/home/zhaoyutong/miniconda3/envs/cuda_ws/bin/nvcc}" \
    cmake -S "${repo_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
fi

cmake --build "${build_dir}" --target gemm_benchmark

"${build_dir}/gemm_benchmark" \
  --sizes "${sizes}" \
  --warmup "${warmup}" \
  --repeat "${repeat}" \
  --device "${device}" \
  --csv "${csv_path}"

${python_bin} "${repo_dir}/scripts/plot_gemm_benchmark.py" \
  --csv "${csv_path}" \
  --output "${results_dir}/gemm_benchmark_summary.png"
