# Reuse Rules and Trust Matrix

## Code layer

### Directly reusable

- `envs/`
- `tasks/`
- `fields/`
- `policies/`
- `algorithms/`
- `utils/`
- `scripts/`

Preconditions:

- no later code change has invalidated the assumption
- future agents preserve the existing observation / action / output contracts

### Reuse rules

- every new training script must keep `outputs/training/<algorithm>/<timestamp>/<run_name>/`
- model weights must live under `checkpoints/`
- training-start snapshots must live under `snapshot/`
- TensorBoard logs must live under `tensorboard/`
- recordings must live under `media/` or `final_eval_media/`
- the canonical zero-spatial-field ablation name is `no_spatial_field`; `task_id_only` is compatibility-only and should not be described as a pure task-id baseline
- for PPO, `eval_reward` and `eval_success_rate` should be interpreted as overall summaries; per-task health lives in fields like `eval_goal_nav_*`, `eval_coverage_*`, and final `eval_metrics.csv` per-task rows
- for variable-`N` PPO, monitor per-`N` keys such as `eval_N4_*`, `eval_N8_*`, ...; the plain `eval_reward` alias is only the cross-`N` overall mean
- for Ubuntu Docker server training on PyTorch images that already include CUDA-enabled torch, install `requirements-server.txt`, not `requirements.txt`, to avoid pip upgrading torch
- server training should remain headless by default; use `MPLBACKEND=Agg` and avoid `--no-headless` for long runs
- TensorBoard should read the canonical training root: `outputs/training`
- server media and checkpoints should continue using the existing run layout under `outputs/training/<algorithm>/<timestamp>/<run_name>/`
- for long Linux all-task PPO training, prefer `bash scripts/run_ppo_all_tasks_long.sh`; edit the top-level `DEFAULT_*` block for persistent run settings, or override `TOTAL_UPDATES`, `CUDA_VISIBLE_DEVICES`, `AGENT_COUNTS`, `EVAL_EPISODES`, or `RECORD_INTERVAL` from the shell for one-off runs
- for sequential PPO task-combination sweeps, prefer `bash scripts/run_ppo_task_queue.sh`; edit the queue rows at the top of the file so every task combination can carry its own training/evaluation/GIF/GPU settings
- for PPO comparisons between single-task specialist policies and multi-task policies, prefer `bash scripts/run_ppo_multitask_suite.sh`; keep the four specialist rows enabled unless intentionally running an ablation
- use `NOTIFY_ONLY=1` with queue scripts to test email notification before long runs; a container needs either SMTP env vars or local `mail` / `mailx` / `sendmail`
- for QQ/Foxmail notifications, copy `configs/examples/wayffusion_mail.env.example` to `.secrets/wayffusion_mail.env` and store the SMTP authorization code there, never in a tracked script
- PPO run directories now use compact task tags and omit `obs_variant`; use `snapshot/cli_args.yaml`, `snapshot/env_config.yaml`, and metrics CSV metadata for full provenance
- PPO queue scripts group child runs as `outputs/training/ppo/<queue_timestamp>/<run_label>/...` by passing `--run_timestamp` and `--run_name` to `train_ppo.py`
- `SyncEnvBatch` remains the serial debug baseline and the default training backend
- `ThreadEnvBatch` is the threaded rollout backend for ordinary multi-env batches; opt in with `--env_backend thread`
- task-balanced rollout batches should be built with `--envs_per_task <K>` when a training run needs fixed per-task environment counts rather than stochastic task sampling
- large-scale diffusion/data-collection style runs should prefer task-balanced thread batches after a sync smoke check, while visualization, GIF debugging, and human-render diagnosis should keep `--env_backend sync`
- PPO queue scripts should leave per-row GPU fields empty when launch-time GPU selection is desired; then `CUDA_VISIBLE_DEVICES=<id> bash scripts/run_ppo_multitask_suite.sh` selects the physical GPU for the whole queue
- non-empty per-row GPU fields intentionally override the inherited `CUDA_VISIBLE_DEVICES` and should only be used when different queue rows must target different devices

## Config layer

### Reusable with caution

- `configs/env/base.yaml`
- `configs/env/multitask.yaml`
- `configs/policy/ppo_cnn_deepsets.yaml`
- `configs/policy/ppo_cnn_deepsets_multitask_20k.yaml`
- `docs/config_reference_zh.md`
- `configs/examples/`

### Drift to watch

- `mlp_ppo.yaml` vs `ppo_mlp.yaml`
- `cnn_deepsets_ppo.yaml` vs `ppo_cnn_deepsets.yaml`

Rule:

- new docs and new scripts should prefer one primary naming scheme
- if aliases remain, they should be marked as legacy or compatibility entries
- new experiments should prefer cloning from `configs/examples/` before editing task- or paper-specific run configs in place
- server deployment docs and scripts should live in `docs/server_training_zh.md` and `scripts/server/` unless a future deployment target needs a separate contract

## Results layer

### Safe to reuse directly

- current smoke-run directory structure
- current test-pass fact
- structured CSVs in `outputs/eval/` that match the latest code contract

### Must be verified before reuse

- summary markdown files
- older benchmark or generalization summaries
- any table that does not clearly expose seed, budget, and checkpoint provenance

## Docs layer

### Trusted

- `README.md`
- `docs/benchmark_spec.md`
- `docs/learning_baselines.md`
- `docs/scaling_experiments.md`
- `docs/reward_normalization.md`
- `docs/variable_agent_policy.md`

### Stale until refreshed

- `outputs/verification.md`

## Mandatory maintenance rule

Future agents must update `agent_memory/` whenever they change:

- code behavior
- configs
- CLI flags
- output directory contracts
- experiment workflow
- docs that redefine the current source of truth

Minimum synchronization steps:

1. update at least one relevant memory card
2. append the change to `cards/03_recent_modifications.md`
3. update `cards/07_maintenance_protocol.md` if the process contract changed
4. keep `manifest.yaml` aligned if cards are added, renamed, or removed

## Operating sequence for future agents

1. read `agent_memory/manifest.yaml`
2. read `audit_scope`, `training_pipeline`, and `audit_findings`
3. confirm the checkpoint path lives under `checkpoints/`
4. prefer the most recent timestamped run for reproduction
5. do not cite smoke-level summaries as if they were final paper evidence
