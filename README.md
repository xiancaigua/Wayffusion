# Centralized Task-Field UAV Benchmark

This project is a lightweight numerical benchmark for centralized multi-UAV multi-task reinforcement learning. The benchmark treats the full UAV swarm as a single agent: the policy consumes a global task field, global agent states, and a task identifier, then emits a joint waypoint action for all UAVs.

The focus is not realistic flight control, MARL, PX4, ROS, or sim2real. The focus is a clean first-stage benchmark for validating:

- multi-channel task fields as a unified task representation,
- a shared waypoint/subgoal action interface,
- centralized joint control over a UAV team,
- multi-task behavior across navigation, coverage, formation, and risk-aware tasks.

## Environment setup

Use the dedicated virtual environment in this workspace:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If you prefer to avoid activation, every command below also works as:

```powershell
.\.venv\Scripts\python.exe <script> ...
```

## Project layout

- `envs/`: centralized Gymnasium-style environment, dynamics, collisions, rewards, metrics
- `tasks/`: four task families plus task sampler
- `fields/`: fixed-channel task field construction and visualization
- `baselines/`: random and heuristic waypoint baselines
- `policies/`: `MLP`, `CNN + DeepSets`, and `CNN + Attention` policies
- `algorithms/`: PPO, SAC, TD3, and BC trainers
- `agent_memory/`: durable audit memories, trust judgments, and a lightweight loader for future agents
- `scripts/check/`: smoke-check rollouts, live visualization, and reward diagnostics
- `scripts/`: dataset generation, training, evaluation, and scaling experiments
- `docs/`: benchmark spec, learning baselines, scaling, reward normalization, and evaluation protocol
- `tests/`: smoke tests for env, fields, and baselines

## Quick start

Generate task fields and one heuristic rollout per task:

```powershell
python scripts/check/generate_task_fields.py --config configs/env/multitask.yaml
```

Run a single rollout with a selected baseline:

```powershell
python scripts/check/run_heuristic_rollout.py --policy random --task goal_nav
python scripts/check/run_heuristic_rollout.py --policy heuristic --task coverage
```

Run a live baseline rollout visualization:

```powershell
python scripts/check/live_rollout.py --task goal_nav --policy heuristic
python scripts/check/live_rollout.py --task coverage --policy random --delay 0.03
```

Evaluate heuristic vs random baselines:

```powershell
python scripts/evaluate_baselines.py --config configs/eval/eval_single_task.yaml
python scripts/evaluate_baselines.py --config configs/eval/eval_multitask.yaml
python scripts/evaluate_baselines.py --config configs/eval/eval_generalization.yaml
```

Diagnose rewards:

```powershell
.\.venv\Scripts\python.exe scripts/check/diagnose_rewards.py --tasks all --agent_counts 4 8 10 20 40 80 100
```

Generate expert datasets:

```powershell
.\.venv\Scripts\python.exe scripts/generate_expert_dataset.py --tasks goal_nav coverage --agent_counts 4 8 10 --episodes 1000
```

Training command convention:

- all training scripts use `--config` for the policy/trainer config, plus `--tasks` and `--agent_counts` to define the training slice
- use `--env-config` only when you want a non-default environment yaml, and use `--obs_variant` / `--scaling_mode` only when you are running ablations or scaling experiments
- all training outputs now go under `outputs/training/<algorithm>/<timestamp>/<run_name>/`; the old `outputs/train/` directory has been merged into `outputs/training/ppo/legacy_*`
- `scripts/train_ppo.py` supports both `--total_updates` and `--target_episodes`; if you want classic PPO stopping, pass `--target_episodes 0` and control training with `--total_updates`
- training-time visualization is evaluation-only: use `--no-headless` to open live rollout windows during periodic eval, and keep `--headless` for normal throughput
- use `--record_eval_episodes <K>` to save the first `K` eval episodes each eval pass, plus `--record_format gif|mp4`, `--record_fps`, and `--record_interval` to control periodic media capture
- TensorBoard is enabled by default on all training scripts; event files are written to `outputs/training/.../tensorboard/`
- use `--console_log_interval` to control live stdout feedback on the trainer's native progress axis: PPO updates, SAC/TD3 environment steps, BC epochs

Train BC:

```powershell
.\.venv\Scripts\python.exe scripts/train_bc.py --config configs/policy/bc_cnn_deepsets.yaml --tasks goal_nav coverage --agent_counts 4 8 10
```

Switch observation variants for ablations with:

```powershell
.\.venv\Scripts\python.exe scripts/train_ppo.py --config configs/policy/ppo_cnn_deepsets.yaml --tasks goal_nav coverage --agent_counts 20 --scaling_mode density_preserving --obs_variant task_id_only
.\.venv\Scripts\python.exe scripts/train_ppo.py --config configs/policy/ppo_cnn_deepsets.yaml --tasks goal_nav coverage --agent_counts 20 --scaling_mode density_preserving --obs_variant single_channel_field
.\.venv\Scripts\python.exe scripts/train_ppo.py --config configs/policy/ppo_cnn_deepsets.yaml --tasks goal_nav coverage --agent_counts 20 --scaling_mode density_preserving --obs_variant multi_channel_field+task_id
```

Train PPO from scratch:

```powershell
.\.venv\Scripts\python.exe scripts/train_ppo.py --config configs/policy/ppo_mlp.yaml --tasks goal_nav coverage --agent_counts 4
.\.venv\Scripts\python.exe scripts/train_ppo.py --config configs/policy/ppo_cnn_deepsets.yaml --tasks goal_nav coverage --agent_counts 4 8 10
.\.venv\Scripts\python.exe scripts/train_ppo.py --config configs/policy/ppo_cnn_deepsets.yaml --tasks goal_nav coverage --agent_counts 4 --no-headless --record_eval_episodes 1 --record_format gif
.\.venv\Scripts\python.exe scripts/train_ppo.py --config configs/policy/ppo_cnn_deepsets.yaml --tasks goal_nav coverage --agent_counts 4 --tensorboard --console_log_interval 5
.\.venv\Scripts\python.exe scripts/train_ppo.py --config configs/policy/ppo_cnn_deepsets_multitask_20k.yaml --tasks goal_nav coverage formation risk_nav --agent_counts 4 --total_updates 850 --target_episodes 0
.\scripts\run_multitask_ppo_20k.ps1
```

View TensorBoard for training runs:

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir outputs/training
```

Train BC + PPO:

```powershell
.\.venv\Scripts\python.exe scripts/train_ppo.py --config configs/policy/ppo_from_bc.yaml --tasks goal_nav coverage --agent_counts 4 8 10 --init_checkpoint outputs/training/bc/<timestamp>/bc_cnn_deepsets_goal_nav_coverage_N4_8_10/checkpoints/checkpoint_0010.pt
```

Train SAC / TD3:

```powershell
.\.venv\Scripts\python.exe scripts/train_sac.py --config configs/policy/sac_cnn_deepsets.yaml --tasks goal_nav coverage --agent_counts 4
.\.venv\Scripts\python.exe scripts/train_td3.py --config configs/policy/td3_cnn_deepsets.yaml --tasks goal_nav coverage --agent_counts 4
```

Evaluate a saved PPO checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_policy.py --checkpoint outputs/training/ppo/<timestamp>/ppo_cnn_deepsets_goal_nav_coverage_N4_8_10/checkpoints/checkpoint_0008.pt --policy-config configs/policy/ppo_cnn_deepsets.yaml --tasks goal_nav coverage --agent_counts 4 8 10
```

Run cross-`N` scaling evaluation:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_scaling.py --checkpoint outputs/training/bc_ppo/<timestamp>/bc_ppo_goal_nav_coverage_N4_8_10/checkpoints/checkpoint_0008.pt --policy-config configs/policy/ppo_from_bc.yaml --tasks goal_nav coverage --agent_counts 4 8 10 20 40 80 100 --protocol variable_N --train_agent_counts 4 8 10 --output-path outputs/eval/scaling_variable_N.csv
```

Run algorithm comparison:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_algorithms.py --configs heuristic random bc ppo bc_ppo sac td3 --tasks goal_nav coverage --agent_counts 4 8 10 --episodes 4 --output-path outputs/eval/algorithm_comparison.csv
```

Run tests:

```powershell
pytest tests/
```

## Observation and action summary

- Observation keys:
  - `task_field`: `[C, H, W]`, with fixed channel order
  - `agents`: `[N, 6]` containing `[x, y, vx, vy, battery, role_id]`
  - `task_id`: one-hot `[4]`
  - `global_info`: `[5]`
- Action:
  - `joint waypoint delta`: `[N, 2]` in `[-1, 1]`
  - environment scales by `max_waypoint_step` and executes via a simple proportional controller

## Outputs

- `outputs/smoke/sanity/task_fields/`: smoke-check field channel visualizations
- `outputs/smoke/sanity/trajectories/`: smoke-check rollout plots
- `outputs/smoke/live/`: optional final-frame captures from live rollout visualization
- `outputs/smoke/diagnostics/`: smoke-check reward diagnostics and component plots
- `outputs/datasets/`: heuristic expert datasets
- `outputs/training/<algorithm>/<timestamp>/<run_name>/`: BC, PPO, SAC, TD3, and BC+PPO training curves/checkpoints
- `outputs/training/.../checkpoints/`: saved model weights and periodic/final checkpoints
- `outputs/training/.../snapshot/`: training-start snapshot with initial model weights plus config / CLI metadata
- `outputs/training/.../tensorboard/`: TensorBoard event files for live training curves and final eval summaries
- `outputs/training/.../media/`: periodic training-eval GIF/MP4 captures when `--record_eval_episodes` is enabled
- `outputs/training/.../final_eval_media/`: final evaluation GIF/MP4 captures for BC / PPO / SAC / TD3 runs when recording is enabled
- `outputs/eval/...`: policy, scaling, and algorithm-comparison tables

## Normalized score

All learning-baseline evaluations report:

```text
normalized_score = (R - R_random) / (R_heuristic - R_random + eps)
```

where `R_random` and `R_heuristic` are measured on the same task set, scaling mode, and swarm size. For multi-task evaluation, the benchmark now computes these references per task family first and then averages over the sampled episode tasks. If the heuristic underperforms random on a hard slice, the two anchors are sorted before normalization and `reference_order_flipped` marks that case in detailed records.

For report stability, the stored `normalized_score` is clipped to `[-5, 5]`, while the raw ratio remains available internally as `normalized_score_raw`.

For large-scale comparisons, note that:

- `collision_rate` is normalized per agent-step
- `path_length` is reported per agent
- `risk_exposure` is reported per agent
- `fixed_map` still increases task volume with `N`, so large swarms are not rewarded for solving the same tiny task with more agents
- `density_preserving` uses smoother sublinear goal / coverage scaling plus revisit-based coverage fulfillment

## Current benchmark status

The benchmark now supports centralized learning-baseline hardening with reward diagnostics, normalized scoring, expert dataset generation, variable-`N` policies, BC, PPO, BC+PPO, SAC, TD3, and cross-`N` evaluation. The main remaining next-stage items are longer training budgets, richer task combinations beyond the current main `goal_nav + coverage` runs, stronger large-scale reward shaping, and broader ablations for paper-quality sweeps.
