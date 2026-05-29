# Training Pipeline and Artifacts

## Current training entrypoints

Primary training entrypoints:

- `scripts/train_ppo.py`
- `scripts/train_sac.py`
- `scripts/train_td3.py`
- `scripts/train_bc.py`

Helper launcher:

- `scripts/run_multitask_ppo_20k.ps1`

Debug / research helpers:

- `scripts/debug_long/analyze_ppo_run.py`
- `scripts/debug_long/diagnose_goal_nav_policy.py`
- `scripts/debug_long/generate_success_expert_dataset.py`
- `scripts/debug_long/collect_dagger_dataset.py`
- `scripts/debug_long/collect_success_policy_dataset.py`
- `scripts/debug_long/generate_coverage_expert_v2_dataset.py`

## Stop conditions

### PPO

`scripts/train_ppo.py` supports two stop modes:

- `--total_updates`
- `--target_episodes`

For classic PPO update-budget training, pass:

- `--target_episodes 0`
- `--total_updates <K>`

### SAC / TD3 / BC

- `SAC` and `TD3` train on environment steps and use config-driven eval intervals
- `BC` trains on epochs
- `BC` now supports `--init_checkpoint` for warm-starting from a previous policy

## Visualization and recording

Training-time rendering and media capture are evaluation-only by design:

- `--no-headless` enables live evaluation windows
- `--record_eval_episodes <K>` records the first `K` eval episodes per capture point
- `--record_format gif|mp4` controls the output format
- `--record_fps` controls export frame rate
- `--record_interval` controls how often evaluation media is saved

The rollout collector itself is intentionally not rendered step by step because that would collapse training throughput.

## Runtime feedback contract

All four training entrypoints now share the same live feedback contract:

- a `tensorboard/` directory is created under every training run
- scalar metrics are written during training, not only at the end
- periodic console feedback is printed during training
- PPO logs on the update axis
- SAC / TD3 log on the environment-step axis
- BC logs on the epoch axis

## PPO evaluation contract

`scripts/train_ppo.py` no longer treats multi-task evaluation as one mixed sampler average.

Current behavior:

- training can still sample tasks randomly when `task_names` has multiple entries
- periodic PPO evaluation now runs a fixed-task eval pass for every task listed in `--tasks`
- `eval_reward` and `eval_success_rate` are compatibility aliases for the overall multi-task summary
- per-task fields such as `eval_goal_nav_return`, `eval_coverage_success_rate`, and `eval_overall_return` are written into `training_metrics.csv` and TensorBoard
- final PPO eval writes one row per task plus one `task_name=overall` row for every evaluated `num_agents`

The same per-task contract now also applies to:

- `scripts/evaluate_policy.py`
- `scripts/evaluate_scaling.py`
- `SAC` periodic and final evaluation
- `TD3` periodic and final evaluation

For variable-`N` PPO training:

- periodic evaluation now monitors every requested `N`
- per-`N` metrics are flattened into keys such as `eval_N4_goal_nav_return`
- legacy aliases such as `eval_reward` map to the mean overall score across monitored `N`

## Debug-long workflow contract

Long-horizon research debug runs now also use:

- `outputs/debug_long/<timestamp>/`
- per-experiment markdown records such as `00_audit.md`, `01_*.md`, and summary JSON/CSV artifacts
- targeted diagnostics for policy alignment, successful-trajectory BC, and DAgger-style dataset collection

These debug artifacts are auxiliary to training outputs, but they are part of the current maintenance workflow and should be kept in sync with code changes.

Current robustness rule for debug data builders:

- success-dataset and dagger-dataset scripts now write `.npz` and JSON sidecars through temporary files plus atomic rename
- if a connection drop interrupts a run, the final dataset path should remain either valid or absent, not half-written

## Current specialist repair chains

### goal_nav

The current effective repair sequence is:

1. run conservative specialist PPO diagnostics on the standard `configs/env/multitask.yaml` slice
2. train BC on curated success-heavy goal-nav datasets
3. collect DAgger-style learner-state relabels when BC+PPO still drifts
4. warm-start PPO from the latest DAgger BC checkpoint with a tightly clamped `log_std`

Current strongest artifacts:

- DAgger BC:
  - `outputs/training/bc/20260528_051955/debug_bc_goal_nav_success_goal_nav_N4_multi_channel_field_plus_task_id/`
- ultra-strict PPO from that BC checkpoint:
  - `outputs/training/bc_ppo/20260528_goalnav_dagger_finetune_ultra_strict/goalnav_dagger_finetune_ultra_strict/`

### coverage

The current effective exploration path is:

1. run low-entropy controlled PPO from scratch
2. if a promising checkpoint exists, continue from that checkpoint rather than restarting
3. only use heuristic BC as a probe, not as the current preferred mainline, unless it clearly exceeds the PPO controlled baseline

Current strongest artifact:

- controlled PPO:
  - `outputs/training/ppo/20260528_phase2_coverage_controlled/phase2_coverage_controlled/`

Current weak artifact:

- heuristic BC probe:
  - `outputs/training/bc/20260528_072926/debug_bc_coverage_coverage_N4_multi_channel_field_plus_task_id/`

## Output directory contract

All training outputs are rooted at:

`outputs/training/<algorithm>/<timestamp>/<run_name>/`

Sibling directories under each run:

- `checkpoints/`
  - periodic and final model weights
- `snapshot/`
  - training-start snapshot
- `tensorboard/`
  - TensorBoard event files
- `media/`
  - periodic evaluation recordings
- `final_eval_media/`
  - final evaluation recordings

Best-eval checkpoint behavior:

- PPO now also writes `checkpoints/checkpoint_best_eval.pt` when a new best eval result is observed
- `best_eval_summary.json` is written in the run root with the corresponding update, eval success, and eval reward
- the current selection rule is:
  - maximize `eval_success_rate`
  - break ties by `eval_reward`
- `scripts/train_ppo.py` final evaluation now defaults to `--final_eval_source best`, so `eval_metrics.csv` is written from the best-eval checkpoint when available instead of the last weight snapshot

## Snapshot contents

`snapshot/` currently stores:

- `train_config.yaml`
- `env_config.yaml`
- `cli_args.yaml`
- `metadata.yaml`
- `initial_model.pt`

`initial_model.pt` is the parameter snapshot at training start, not the trained result.

## Checkpoint lookup rules

- `latest_checkpoint(...)` performs a recursive lookup for `checkpoint*.pt`
- `evaluate_algorithms.py` already handles checkpoints stored under `checkpoints/`
- `evaluate_policy.py` and `evaluate_scaling.py` still expect an explicit checkpoint path

## Trusted conclusions

- the training / evaluation / media / parameter-saving directory structure is stable
- recording, snapshotting, TensorBoard, and checkpointing are now one coherent engineering chain
- future training scripts should preserve this contract instead of writing weights directly into the run root
