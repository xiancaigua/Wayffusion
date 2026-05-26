#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "usage: $0 <run_label> <tasks> <total_updates> <cuda_visible_devices> [extra train_ppo.py args...]" >&2
  exit 2
fi

RUN_LABEL="$1"
TASKS="$2"
TOTAL_UPDATES="$3"
CUDA_DEVICES="$4"
shift 4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export MPLBACKEND="${MPLBACKEND:-Agg}"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

CONFIG="${CONFIG:-configs/policy/ppo_cnn_deepsets_multitask_20k.yaml}"
ENV_CONFIG="${ENV_CONFIG:-configs/env/multitask.yaml}"
SCALING_MODE="${SCALING_MODE:-fixed_map}"
OBS_VARIANT="${OBS_VARIANT:-multi_channel_field+task_id}"
TARGET_EPISODES="${TARGET_EPISODES:-0}"
AGENT_COUNTS="${AGENT_COUNTS:-4}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
RECORD_EVAL_EPISODES="${RECORD_EVAL_EPISODES:-1}"
RECORD_FORMAT="${RECORD_FORMAT:-gif}"
RECORD_FPS="${RECORD_FPS:-8}"
RECORD_INTERVAL="${RECORD_INTERVAL:-4}"
CONSOLE_LOG_INTERVAL="${CONSOLE_LOG_INTERVAL:-10}"
ENV_BACKEND="${ENV_BACKEND:-thread}"
ENVS_PER_TASK="${ENVS_PER_TASK:-4}"
ENV_WORKERS="${ENV_WORKERS:-16}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

read -r -a task_array <<< "${TASKS}"
read -r -a agent_count_array <<< "${AGENT_COUNTS}"

echo "[ppo-target] label=${RUN_LABEL}"
echo "[ppo-target] tasks=${TASKS}"
echo "[ppo-target] updates=${TOTAL_UPDATES}"
echo "[ppo-target] cuda_visible_devices=${CUDA_DEVICES}"
echo "[ppo-target] python=${PYTHON_BIN}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" scripts/train_ppo.py \
  --config "${CONFIG}" \
  --env-config "${ENV_CONFIG}" \
  --tasks "${task_array[@]}" \
  --agent_counts "${agent_count_array[@]}" \
  --scaling_mode "${SCALING_MODE}" \
  --obs_variant "${OBS_VARIANT}" \
  --total_updates "${TOTAL_UPDATES}" \
  --target_episodes "${TARGET_EPISODES}" \
  --eval_episodes "${EVAL_EPISODES}" \
  --record_eval_episodes "${RECORD_EVAL_EPISODES}" \
  --record_format "${RECORD_FORMAT}" \
  --record_fps "${RECORD_FPS}" \
  --record_interval "${RECORD_INTERVAL}" \
  --console_log_interval "${CONSOLE_LOG_INTERVAL}" \
  --env_backend "${ENV_BACKEND}" \
  --envs_per_task "${ENVS_PER_TASK}" \
  --env_workers "${ENV_WORKERS}" \
  --run_timestamp "${RUN_TIMESTAMP}" \
  --run_name "${RUN_LABEL}" \
  --headless \
  --tensorboard \
  "$@"
