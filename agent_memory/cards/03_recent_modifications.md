# Recent Applied Modifications

## Theme A: Evaluation-time visualization and recording

Implemented:

- `headless` / `no-headless` toggles on training entrypoints
- shared evaluation recording for `PPO`, `SAC`, `TD3`, and `BC`
- unified `gif/mp4` export through `utils/evaluation.py`
- `eval_media_dir` tracking in evaluation outputs

Design choice:

- live rendering and recording are limited to evaluation episodes
- the rollout collector is not rendered frame by frame

## Theme B: Stable render path for media generation

Implemented:

- `rgb_array` rendering uses a NumPy image path
- `human` mode keeps interactive visualization

Reason:

- recording should not depend on a GUI backend
- the previous matplotlib backend path was fragile on some local runs

## Theme C: Timestamped training output roots

Implemented:

- all training outputs now go to `outputs/training/<algorithm>/<timestamp>/<run_name>/`
- the timestamp layer is built through a shared helper

Reason:

- repeated runs no longer overwrite each other
- per-algorithm runs are easier to audit and compare

## Theme D: Sibling directories for checkpoints and snapshots

Implemented:

- model weights are stored under `checkpoints/`
- training-start material is stored under `snapshot/`

Reason:

- start state and trained state are now explicitly separated
- `media/`, `checkpoints/`, `snapshot/`, and `tensorboard/` live at the same level

## Theme E: PPO stop contract

Implemented:

- `scripts/train_ppo.py` supports both `--total_updates` and `--target_episodes`
- the multitask PPO launcher defaults to update-based stopping

Reason:

- supports classic PPO training by update count
- still keeps episode-budget control when needed

## Theme F: Multitask PPO launcher

Implemented:

- `configs/policy/ppo_cnn_deepsets_multitask_20k.yaml`
- `scripts/run_multitask_ppo_20k.ps1`

Current default:

- task interleaving over `goal_nav coverage formation risk_nav`
- centralized PPO
- `CNN + DeepSets`
- update-based stopping

## Theme G: Expanded tests

Added coverage for:

- media export
- non-headless human render path
- timestamped training paths
- snapshot and recursive checkpoint lookup
- PPO target-episode early stop
- agent memory module loading

## Theme H: TensorBoard and live console feedback

Implemented:

- shared TensorBoard event writing for `PPO`, `SAC`, `TD3`, and `BC`
- periodic console progress lines during training
- final evaluation summaries written to TensorBoard
- per-run `tensorboard/` directory under every training output

Design choice:

- TensorBoard is attached to the same run root as checkpoints and media
- console logging is emitted on the trainer's native progress axis: update, step, or epoch

## Theme I: Agent memory synchronization rule

Implemented:

- `agent_memory/README.md` now states a mandatory synchronization rule
- new `maintenance_protocol` card formalizes the process contract

Rule:

- every code, config, doc, CLI, or output-contract change must update `agent_memory/` in the same task

## Trusted conclusion

These changes are no longer isolated patches. They now exist consistently in:

- code
- docs
- tests
- output contract
- operational memory

That combination should be treated as the current engineering source of truth.
