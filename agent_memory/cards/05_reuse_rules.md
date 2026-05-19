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
