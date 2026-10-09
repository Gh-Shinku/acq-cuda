#!/usr/bin/env bash
#
# Profile one of the hand-written FP16 GEMM kernels with Nsight Compute.
#
# Usage:
#   tools/scripts/profile_hgemm.sh [kernel]      # default: mma_hgemm
#
# Examples:
#   tools/scripts/profile_hgemm.sh                # the mma.sync kernel
#   tools/scripts/profile_hgemm.sh wmma_hgemm     # the nvcuda::wmma kernel
#   tools/scripts/profile_hgemm.sh custom_hgemm   # the SIMT kernel
#   KERNEL=mma_hgemm SECTIONS=--set=detailed tools/scripts/profile_hgemm.sh
#
# Environment overrides:
#   NCU       path to ncu (default: the one in PATH, else the conda env)
#   BENCH     benchmark binary (default: build/hgemm_benchmark)
#   SECTIONS  ncu section/flag list (default: a focused set)
#   SKIP      launches to skip before profiling (default: 8)
#   COUNT     launches to profile (default: 1)
#   SUDO      1 to run ncu via sudo (needed when counters are restricted)
#
# IMPORTANT: run this while the GPUs are idle. If another workload is using the
# device the numbers are meaningless (times can swing by more than 2x).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KERNEL="${1:-${KERNEL:-mma_hgemm}}"

if [[ -n "${NCU:-}" ]]; then
  ncu_bin="${NCU}"
elif command -v ncu >/dev/null 2>&1; then
  ncu_bin="$(command -v ncu)"
elif [[ -x /home/zhaoyutong/miniconda3/envs/cuda_ws/bin/ncu ]]; then
  ncu_bin=/home/zhaoyutong/miniconda3/envs/cuda_ws/bin/ncu
else
  echo "error: ncu not found; set NCU=/path/to/ncu" >&2
  exit 1
fi

BENCH="${BENCH:-${REPO_ROOT}/build/hgemm_benchmark}"
if [[ ! -x "${BENCH}" ]]; then
  echo "error: benchmark not found at ${BENCH}" >&2
  echo "       build it with: cmake --build build --target hgemm_benchmark -j" >&2
  exit 1
fi

SECTIONS="${SECTIONS:---section SpeedOfLight --section Occupancy \
--section SchedulerStats --section WarpStateStats \
--section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
--section InstructionStats --section LaunchStats}"

# Explicit warp-stall breakdown; --page details does not print all of these.
METRICS="${METRICS:-smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_selected_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_membar_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_drain_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_misc_per_issue_active.ratio,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_lsu_cycles_active.avg.pct_of_peak_sustained_elapsed,\
smsp__inst_executed_pipe_tensor.sum,\
smsp__inst_executed_pipe_lsu.sum,\
smsp__inst_executed_pipe_alu.sum,\
smsp__inst_executed_pipe_fma.sum}"

# Allow disabling the metrics list by setting METRICS= (empty).
metrics_flag=()
if [[ -n "${METRICS}" ]]; then
  metrics_flag=(--metrics "${METRICS}")
fi
SKIP="${SKIP:-8}"
COUNT="${COUNT:-1}"

report_dir="${REPO_ROOT}/ncu_reports"
mkdir -p "${report_dir}"
stamp="$(date +%Y%m%d_%H%M%S)"
report="${report_dir}/${KERNEL}_${stamp}"

# Warn if the GPUs are busy.
if command -v nvidia-smi >/dev/null 2>&1; then
  busy="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null |
          awk '$1 > 20 {c++} END {print c+0}')"
  if [[ "${busy}" -gt 0 ]]; then
    echo "warning: ${busy} GPU(s) look busy (>20% util); profiling numbers may be invalid." >&2
  fi
fi

run_ncu() {
  # Use a token-boundary regex: a plain 'mma_hgemm' would also match
  # 'wmma_hgemm', and the previous run profiled the wrong kernel because of it.
  # shellcheck disable=SC2086
  "${ncu_cmd[@]}" \
    --kernel-name-base demangled \
    --kernel-name "regex:(^|[^A-Za-z0-9_])${KERNEL}([^A-Za-z0-9_]|$)" \
    --launch-skip "${SKIP}" --launch-count "${COUNT}" \
    --target-processes all \
    ${SECTIONS} \
    "${metrics_flag[@]}" \
    --print-summary per-kernel \
    -o "${report}" \
    "${BENCH}" --warmup "$((SKIP + 2))" --samples 1 --launches-per-sample 1 \
    --csv /dev/null
}

ncu_cmd=("${ncu_bin}")
if [[ "${SUDO:-0}" == "1" ]]; then
  ncu_cmd=(sudo "${ncu_bin}")
fi

echo "profiling kernel '${KERNEL}' with ${ncu_bin}"
echo "report: ${report}.ncu-rep"
if [[ "${SUDO:-0}" == "1" ]]; then
  echo "note: running ncu via sudo; the report will be owned by root."
fi
echo

set +e
run_ncu
status=$?
set -e

if [[ ${status} -ne 0 ]]; then
  cat >&2 <<EOF

ncu exited with status ${status}.

If the output mentions ERR_NVGPUCTRPERM, performance counters are restricted.
Either rerun with sudo:

  SUDO=1 tools/scripts/profile_hgemm.sh ${KERNEL}

or enable them permanently (requires root + reboot):

  sudo tools/scripts/enable-ncu-permissions.sh
EOF
  exit ${status}
fi

echo
echo "done. open the report with:"
echo "  ${ncu_bin%-ncu}-ui ${report}.ncu-rep   # if ncu-ui is installed"
echo "  ${ncu_bin} --import ${report}.ncu-rep --page details"
