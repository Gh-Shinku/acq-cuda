#!/usr/bin/env bash
#
# Allow non-root users to collect NVIDIA GPU performance counters, which Nsight
# Compute / ncu needs. Without this ncu fails with:
#
#   ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
#   Performance Counters on the target device 0.
#
# Run this once with sudo. It writes a modprobe option; the change takes effect
# after a reboot (or after reloading the nvidia kernel modules, which is
# disruptive on a busy machine).
#
#   sudo tools/scripts/enable-ncu-permissions.sh
#
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "error: must run as root: sudo $0" >&2
  exit 1
fi

conf=/etc/modprobe.d/nvidia-profiler.conf
line='options nvidia NVreg_RestrictProfilingToAdminUsers=0'

if [[ -f "${conf}" ]] && grep -qF "${line}" "${conf}"; then
  echo "already configured: ${conf}"
else
  echo "${line}" > "${conf}"
  echo "wrote ${conf}:"
  echo "  ${line}"
fi

if [[ -f /proc/driver/nvidia/params ]] &&
   grep -q '^RestrictProfilingToAdminUsers: 0' /proc/driver/nvidia/params; then
  echo "profiling is already unrestricted on the running driver."
  exit 0
fi

cat <<'EOF'

Now reboot (recommended), or reload the driver modules once the GPUs are idle:

  sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
  sudo modprobe nvidia

Verify with:

  cat /proc/driver/nvidia/params | grep RestrictProfilingToAdminUsers

It should print: RestrictProfilingToAdminUsers: 0

Note: running ncu itself as root (sudo ncu ...) also bypasses the restriction
and needs no reboot.
EOF
