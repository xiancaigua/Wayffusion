#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-${ALL4_CUDA_VISIBLE_DEVICES:-1}}"
"${SCRIPT_DIR}/_run_ppo_target.sh" "multi_all4" "goal_nav coverage formation risk_nav" "${TOTAL_UPDATES:-25000}" "${CUDA_DEVICES}" "$@"
