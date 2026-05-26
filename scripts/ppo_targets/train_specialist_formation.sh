#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-${FORMATION_CUDA_VISIBLE_DEVICES:-3}}"
"${SCRIPT_DIR}/_run_ppo_target.sh" "specialist_formation" "formation" "${TOTAL_UPDATES:-20000}" "${CUDA_DEVICES}" "$@"
