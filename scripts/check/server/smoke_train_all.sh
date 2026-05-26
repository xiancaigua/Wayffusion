#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

TASKS=(${TASKS:-goal_nav coverage})
AGENT_COUNTS=(${AGENT_COUNTS:-4})
EVAL_EPISODES="${EVAL_EPISODES:-1}"
DATASET_PATH="${DATASET_PATH:-outputs/datasets/server_smoke_expert_goal_nav_coverage_N4.npz}"

echo "[server-smoke] python=${PYTHON_BIN}"
echo "[server-smoke] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"

"${PYTHON_BIN}" scripts/check/server/check_server_env.py
"${PYTHON_BIN}" -m pytest -q tests/

"${PYTHON_BIN}" scripts/generate_expert_dataset.py \
  --tasks "${TASKS[@]}" \
  --agent_counts "${AGENT_COUNTS[@]}" \
  --episodes 1 \
  --output "${DATASET_PATH}"

"${PYTHON_BIN}" scripts/train_ppo.py \
  --config configs/policy/ppo_server_smoke.yaml \
  --tasks "${TASKS[@]}" \
  --agent_counts "${AGENT_COUNTS[@]}" \
  --target_episodes 0 \
  --eval_episodes "${EVAL_EPISODES}" \
  --console_log_interval 1 \
  --tensorboard \
  --headless

"${PYTHON_BIN}" scripts/train_sac.py \
  --config configs/policy/sac_server_smoke.yaml \
  --tasks "${TASKS[@]}" \
  --agent_counts "${AGENT_COUNTS[@]}" \
  --eval_episodes "${EVAL_EPISODES}" \
  --console_log_interval 16 \
  --tensorboard \
  --headless

"${PYTHON_BIN}" scripts/train_td3.py \
  --config configs/policy/td3_server_smoke.yaml \
  --tasks "${TASKS[@]}" \
  --agent_counts "${AGENT_COUNTS[@]}" \
  --eval_episodes "${EVAL_EPISODES}" \
  --console_log_interval 16 \
  --tensorboard \
  --headless

"${PYTHON_BIN}" scripts/train_bc.py \
  --config configs/policy/bc_server_smoke.yaml \
  --tasks "${TASKS[@]}" \
  --agent_counts "${AGENT_COUNTS[@]}" \
  --dataset "${DATASET_PATH}" \
  --eval_episodes "${EVAL_EPISODES}" \
  --console_log_interval 1 \
  --tensorboard \
  --headless

echo "[server-smoke] all algorithm smoke checks passed"
