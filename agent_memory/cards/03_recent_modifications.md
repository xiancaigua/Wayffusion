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

## Theme J: Core credibility repair for scaling, policy math, and ablation semantics

Implemented:

- `waypoint_controller(...)` now clips against `map_size`, not a hard-coded unit square
- `density_preserving` runs can now move into coordinates larger than `1.0`
- `MLP`, `CNN + DeepSets`, and `Attention` policies now use tanh-squashed Gaussian action sampling with corrected log-prob evaluation
- formation radius mismatch is now represented as a positive error with an explicit negative penalty term
- canonical ablation naming is now `no_spatial_field`, while `task_id_only` remains a deprecated alias
- reference normalization now marks unstable anchors and preserves raw-return / anchor metadata more explicitly

Reason:

- these were benchmark-trust issues, not cosmetic cleanups
- cross-`N` evaluation and policy-gradient math are now aligned with the intended centralized benchmark contract

## Trusted conclusion

These changes are no longer isolated patches. They now exist consistently in:

- code
- docs
- tests
- output contract
- operational memory

That combination should be treated as the current engineering source of truth.

## Theme K: Chinese config reference and template config directory

Implemented:

- `docs/config_reference_zh.md` was refreshed into a script-oriented Chinese config guide
- `configs/examples/` now contains ready-to-copy template files for env, policy, and eval categories
- a lightweight parser test now guards the template directory against drift

Reason:

- users now have a single Chinese entrypoint for understanding config categories
- new experiments can start from stable templates instead of mutating old run configs blindly

## Theme L: PPO evaluation now runs per-task instead of only mixed-sampler averages

Implemented:

- `utils/evaluation.py` now provides a fixed-task `evaluate_policy_per_task(...)` helper
- PPO periodic evaluation now records per-task metrics plus an overall summary
- PPO final evaluation now writes per-task rows and an `overall` row into `eval_metrics.csv`
- TensorBoard final-eval namespaces now include per-task paths such as `ppo/final_eval/N4/goal_nav/...`

Reason:

- a multi-task training run should not hide a collapsed task behind one mixed average
- per-task regressions are now visible during training and in saved evaluation tables

## Theme M: Per-task evaluation propagated to eval scripts and off-policy trainers

Implemented:

- `scripts/evaluate_policy.py` now writes per-task rows plus an `overall` row
- `scripts/evaluate_scaling.py` now writes per-task rows plus an `overall` row for every evaluated `N`
- `SAC` and `TD3` periodic evaluation now flatten per-task metrics into training records
- `SAC` and `TD3` final eval now mirror PPO with per-task TensorBoard namespaces and per-task `eval_metrics.csv`
- variable-`N` PPO periodic evaluation now monitors all requested `N`, not just the first one
- restored minimal default configs for `configs/policy/sac_cnn_deepsets.yaml` and `configs/policy/td3_cnn_deepsets.yaml` so the touched training scripts keep a valid default entrypoint

Reason:

- trust-worthy evaluation should expose both task-level failures and cross-`N` failures
- fixed-task breakdowns are now consistent across online training logs, eval scripts, and saved CSV outputs

## Theme N: Ubuntu Docker server training adaptation

Implemented:

- added `requirements-server.txt` for PyTorch Docker images that already provide CUDA-enabled torch
- kept the existing `requirements.txt` as the generic environment file and added the missing `psutil` dependency there
- added `docs/server_training_zh.md` with Docker launch, proxy verification, dependency installation, GPU checks, smoke tests, TensorBoard forwarding, long-run training, and common troubleshooting
- added `scripts/server/check_server_env.py` for offline server environment diagnostics
- added `scripts/server/smoke_train_ppo.sh` for a short headless PPO training and GIF-recording check
- added `scripts/server/start_tensorboard.sh` for the canonical `outputs/training` TensorBoard logdir
- added a README entry pointing server users to the dedicated guide
- extended `.gitignore` so server-side test runs do not leave Python cache files as untracked worktree noise

Reason:

- the target server image is `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel`, so server installs must avoid upgrading torch through pip
- server training must remain headless and preserve the existing `outputs/training/<algorithm>/<timestamp>/<run_name>/` artifact layout
- `utils/profiling.py` imports `psutil`, so server and generic dependency files now declare it explicitly

## Theme O: Long all-task PPO training launcher

Implemented:

- added `scripts/run_ppo_all_tasks_long.sh` as the Linux/Ubuntu launcher for long PPO training across `goal_nav coverage formation risk_nav`
- default run uses `configs/policy/ppo_cnn_deepsets_multitask_20k.yaml`, `total_updates=2000`, `target_episodes=0`, `eval_episodes=5`, and `agent_counts=4`
- periodic evaluation and checkpointing follow the config's PPO `eval_interval` of 25 updates
- GIF recording is enabled with `record_eval_episodes=1` and `record_interval=4`, so media is saved every 100 updates plus final evaluation media
- script defaults to headless training, TensorBoard enabled, `MPLBACKEND=Agg`, and `CUDA_VISIBLE_DEVICES=0`, while allowing shell environment overrides
- script has a top-level editable `DEFAULT_*` configuration block so normal parameter changes can be made inside the `.sh` file
- README now points Linux users to the long all-task launcher

Reason:

- server users need one direct command for long all-task PPO training that preserves the existing artifact contract
- checkpoints, TensorBoard logs, evaluation CSVs, and GIFs continue to live under `outputs/training/ppo/<timestamp>/<run_name>/`

## Theme P: Sequential PPO task-combination queue launcher

Implemented:

- added `scripts/run_ppo_task_queue.sh` for sequential PPO training over editable task-combination rows
- each queue row has its own label, task list, agent counts, update budget, eval episodes, GIF recording cadence, GPU selection, and extra CLI args
- queue runs remain headless, use TensorBoard, preserve `outputs/training/ppo/<timestamp>/<run_name>/`, and write queue logs under `outputs/training/task_queue/<timestamp>/`
- completion notification targets `muadib@foxmail.com` by default and supports SMTP env vars or local `mail` / `mailx` / `sendmail`
- README now includes the queue launcher entrypoint

Reason:

- task-combination sweeps need repeatable sequential execution without manually starting every PPO run
- per-row parameters make it practical to debug different task mixes and hyperparameter variants while keeping the existing PPO training contract

## Theme Q: Core code explanatory comments

Implemented:

- added explanatory docstrings and comments to `envs/centralized_env.py` around centralized environment setup, observation/action contracts, runtime scaling, state initialization, task-field composition, transition stepping, collision handling, reward composition, and info metrics
- added comments to `algorithms/ppo.py` explaining rollout storage, reward normalization, critic bootstrap, GAE, minibatch flattening, clipped PPO ratios, and loss components
- added neural-network comments to `policies/cnn_deepsets_policy.py`, `policies/attention_policy.py`, and `policies/mlp_policy.py` describing architecture roles, variable-N handling, token/pooling logic, and centralized value/action heads
- added distribution comments to `policies/action_distribution.py` explaining tanh-squashed Gaussian actions and log-prob change-of-variables correction

Reason:

- future readers need the environment, algorithm, and policy framework to be understandable without reverse-engineering every tensor transformation
- comments document intent and contracts only; no algorithm semantics or training outputs were changed
