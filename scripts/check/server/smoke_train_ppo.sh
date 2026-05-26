#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" scripts/train_ppo.py \
  --config configs/policy/ppo_server_smoke.yaml \
  --tasks goal_nav coverage \
  --agent_counts 4 \
  --target_episodes 0 \
  --eval_episodes 1 \
  --record_eval_episodes 1 \
  --record_format gif \
  --console_log_interval 1
