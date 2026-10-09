#!/usr/bin/env bash
# Run Nsight Compute for the Stream-K kernel when GPU counters require root.
# Usage: tools/scripts/profile_hgemm_streamk.sh
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SUDO="${SUDO:-1}"
export SECTIONS="${SECTIONS:---section SpeedOfLight --section Occupancy --section WarpStateStats --section MemoryWorkloadAnalysis}"
export METRICS="${METRICS:-}"
exec "${script_dir}/profile_hgemm.sh" streamk_hgemm
