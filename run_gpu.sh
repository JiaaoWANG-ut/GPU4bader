#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_gpu.sh [charge_file] [device_type] [device_num]
#
# Examples:
#   ./run_gpu.sh ../NaCl/CHG_NaCl nvidia 0
#   ./run_gpu.sh ../NaCl/CHG_NaCl host
#
# device_type: nvidia | host | multicore

CHG_FILE="${1:-../NaCl/CHG_NaCl}"
DEVICE_TYPE="${2:-nvidia}"
DEVICE_NUM="${3:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NVHPC_VER="${NVHPC_VER:-25.1}"
NVHPC_ROOT="${NVHPC_ROOT:-/opt/nvidia/hpc_sdk/Linux_x86_64/${NVHPC_VER}}"

export LD_LIBRARY_PATH="${NVHPC_ROOT}/compilers/lib:${NVHPC_ROOT}/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export ACC_DEVICE_TYPE="${DEVICE_TYPE}"
export ACC_DEVICE_NUM="${DEVICE_NUM}"
export NVCOMPILER_ACC_TIME="${NVCOMPILER_ACC_TIME:-1}"

echo "[run_gpu] NVHPC_ROOT=${NVHPC_ROOT}"
echo "[run_gpu] ACC_DEVICE_TYPE=${ACC_DEVICE_TYPE}"
echo "[run_gpu] ACC_DEVICE_NUM=${ACC_DEVICE_NUM}"
echo "[run_gpu] CHG_FILE=${CHG_FILE}"

"${SCRIPT_DIR}/bader" "${CHG_FILE}"
