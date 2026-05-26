#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-${RISK_NAV_CUDA_VISIBLE_DEVICES:-0}}"
"${SCRIPT_DIR}/_run_ppo_target.sh" "specialist_risk_nav" "risk_nav" "${TOTAL_UPDATES:-20000}" "${CUDA_DEVICES}" "$@"
