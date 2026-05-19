# Training Pipeline and Artifacts

## Current training entrypoints

Primary training entrypoints:

- `scripts/train_ppo.py`
- `scripts/train_sac.py`
- `scripts/train_td3.py`
- `scripts/train_bc.py`

Helper launcher:

- `scripts/run_multitask_ppo_20k.ps1`

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
