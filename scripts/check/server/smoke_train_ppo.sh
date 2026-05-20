#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export MPLBACKEND="${MPLBACKEND:-Agg}"

python scripts/train_ppo.py \
  --config configs/policy/ppo_cnn_deepsets.yaml \
  --tasks goal_nav coverage \
  --agent_counts 4 \
  --total_updates 2 \
  --target_episodes 0 \
  --eval_episodes 1 \
  --record_eval_episodes 1 \
  --record_format gif \
  --console_log_interval 1
