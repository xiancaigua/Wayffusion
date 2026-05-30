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

## Theme S: goal_nav remaining-goal observability and conservative PPO diagnostics

Implemented:

- `tasks/goal_nav.py` and `tasks/risk_nav.py` now rebuild the `goal_reward` field from only unreached goals each step instead of keeping a static reset-time goal map
- `tests/test_rewards_basic.py` now checks that goal-progress shaping ignores goals already marked as reached
- `algorithms/ppo.py` now clamps `log_std` both at trainer init and after checkpoint load, and logs rollout reward stats, terminal metrics, KL/clip diagnostics, ratio mean, gradient norm, explained variance, and policy std summaries
- `utils/evaluation.py` now accumulates episode-level reward components into eval records for deeper reward audits

Reason:

- goal-nav specialists were repeatedly re-attracted to already completed goals
- conservative PPO fine-tunes need explicit `log_std` enforcement after loading BC checkpoints
- long-horizon debug work needed richer diagnostics than `eval_reward` / `eval_success_rate` alone

## Theme T: goal_nav specialist repair via success-trajectory BC, DAgger BC, and ultra-strict PPO

Implemented:

- added goal-nav debug configs:
  - `configs/policy/debug_bc_goal_nav.yaml`
  - `configs/policy/debug_bc_goal_nav_success.yaml`
  - `configs/policy/debug_ppo_goal_nav_controlled.yaml`
  - `configs/policy/debug_ppo_goal_nav_bc_finetune.yaml`
  - `configs/policy/debug_ppo_goal_nav_bc_finetune_strict.yaml`
  - `configs/policy/debug_ppo_goal_nav_dagger_finetune_ultra_strict.yaml`
- `scripts/train_bc.py` now accepts `--init_checkpoint` so BC can be warm-started from previous BC or PPO checkpoints during DAgger-style repair loops
- `outputs/debug_long/20260528_035433/` now records the success-only BC run, BC+PPO fine-tune, DAgger dataset collection, and DAgger BC retrain

Current best evidence:

- success-only DAgger BC run:
  - `outputs/training/bc/20260528_051955/debug_bc_goal_nav_success_goal_nav_N4_multi_channel_field_plus_task_id/`
  - final `success_rate_mean=0.8`
  - final `goal_coverage_ratio_mean=0.9125`
  - final `collision_rate_mean=0.0646`
- ultra-strict PPO fine-tune from the DAgger BC checkpoint:
  - `outputs/training/bc_ppo/20260528_goalnav_dagger_finetune_ultra_strict/goalnav_dagger_finetune_ultra_strict/`
  - final `success_rate_mean=0.8`

## Theme AC: coverage centralized per-agent waypoint allocation experiments

Implemented:

- `CNNDeepSetsPolicy` now has an optional `coverage_utility_slot_head`, disabled by default.
- The new head reads the centralized multi-channel field and adds a small per-agent waypoint bias from a pooled coverage utility grid.
- The head keeps the same action contract: output is still `[B, N, 2]`, log-prob remains one joint log-prob per env, and `agent_mask` still zeros padded agents.
- `policies/__init__.py` forwards the new config fields.
- `tests/test_variable_policies.py` now checks the utility slot head and cross-task compatibility.
- New configs:
  - `configs/policy/debug_bc_coverage_utilityslot.yaml`
  - `configs/policy/debug_ppo_coverage_utilityslot_ultra_strict.yaml`

Diagnostic result:

- Direct untrained utility-slot behavior was bad (`coverage_ratio_mean≈0.314`, high collision), so the head is not a standalone heuristic.
- BC on `coverage_success_from_phase24_best_round4.npz` produced:
  - run: `outputs/training/bc/20260530_025202/debug_bc_coverage_utilityslot_coverage_N4_multi_channel_field_plus_task_id/`
  - `success_rate_mean=0.10`
  - `coverage_ratio_mean≈0.666`
  - `collision_rate_mean≈0.000375`
- PPO fine-tune `phase27_coverage_utilityslot` was stopped at update 40 because eval stayed at `success_rate=0.0`.

Conclusion:

- A fixed coverage utility bias can preserve low collision after BC, but it destabilizes PPO fine-tuning.
- Do not continue this exact fixed-bias branch unless the bias is made learnable or much weaker.

## Theme AD: coverage expert-v3 local frontier teacher

Implemented:

- `scripts/debug_long/generate_coverage_expert_v2_dataset.py` now also exposes `CoverageExpertV3` and `--expert v3`.
- `CoverageExpertV3` uses local remaining-demand frontier selection plus short-range collision-avoidance repulsion.
- `scripts/debug_long/generate_success_expert_dataset.py` now supports:
  - `--teacher coverage_expert_v2`
  - `--teacher coverage_expert_v3`

Diagnostic result:

- `CoverageExpertV2` diagnostic: `success=0.0`, `coverage_ratio_mean≈0.538`, high collision.
- local frontier without repulsion: `success≈0.0625`, `coverage_ratio_mean≈0.645`, but collision was too high.
- `CoverageExpertV3` with repulsion: `success≈0.07`, `coverage_ratio_mean≈0.622`, `collision≈0.0041`.

Conclusion:

- `CoverageExpertV3` is not a solved expert, but it is a better source of successful coverage trajectories than `CoverageExpertV2`.
- Next coverage branch should collect success-only v3 trajectories, then train BC/PPO from that dataset.

## Theme AE: coverage permutation-invariant BC breakthrough

Implemented:

- `algorithms/bc.py` now supports optional `permutation_invariant_loss`.
- The loss works in waypoint space, not raw action space:
  - predicted waypoint = current position + predicted action * `bc_waypoint_step`
  - expert waypoint = current position + expert action * `bc_waypoint_step`
  - all target waypoint assignments are enumerated for small N, and the minimum assignment loss is used
- `scripts/train_bc.py` injects `bc_waypoint_step` from the env config into BC config.
- Added `tests/test_bc_permutation_loss.py`.
- Added `configs/policy/debug_bc_coverage_successonly_perm.yaml`.
- Added `scripts/debug_long/merge_expert_datasets.py` for atomic NPZ dataset merges.

Data generated:

- `outputs/debug_long/20260530_coverage_expert_v3/coverage_expert_v3_success_N4_e40.npz`
  - 40 successful episodes
  - 7354 samples
  - 665 attempts
  - mean successful collision rate: `0.0`
- `outputs/debug_long/20260530_coverage_expert_v3/coverage_success_round4_plus_v3_e40.npz`
  - merged samples: 17162
  - inputs:
    - phase24 round4 success data: 9808 samples
    - v3 success data: 7354 samples

Key result:

- permutation-invariant BC run:
  - `outputs/training/bc/20260530_032314/debug_bc_coverage_successonly_perm_coverage_N4_multi_channel_field_plus_task_id/`
  - `success_rate_mean=0.25`
  - `coverage_ratio_mean≈0.717`
  - `collision_rate_mean≈0.0079`
  - `return_mean≈10.61`

Conclusion:

- Coverage BC was materially harmed by fixed-index expert action matching.
- The current best coverage specialist checkpoint is now the permutation-invariant BC checkpoint, pending PPO fine-tune.
  - final `goal_coverage_ratio_mean=0.9425`
  - final `collision_rate_mean=0.0275`
  - final `return_mean=10.1225`

Reason:

- naive PPO from scratch on goal_nav showed nonzero intermediate improvements but repeatedly fell back to `success_rate≈0`
- BC from generic heuristic data improved alignment but not enough robustness
- DAgger-style relabeling on learner states produced the first specialist line that stayed in a high-success region under PPO

## Theme U: coverage specialist controlled-PPO branch and BC probe

Implemented:

- added coverage debug configs:
  - `configs/policy/debug_ppo_coverage_controlled.yaml`
  - `configs/policy/debug_ppo_coverage_bc_finetune.yaml`
  - `configs/policy/debug_bc_coverage.yaml`
  - `configs/policy/debug_ppo_coverage_long_horizon.yaml`
- generated heuristic expert dataset:
  - `outputs/datasets/expert_coverage_N4_phase3_e80.npz`
- trained BC coverage probe:
  - `outputs/training/bc/20260528_072926/debug_bc_coverage_coverage_N4_multi_channel_field_plus_task_id/`

Current status:

- best PPO controlled coverage line so far comes from:
  - `outputs/training/ppo/20260528_phase2_coverage_controlled/phase2_coverage_controlled/`
  - `eval_coverage_coverage_ratio≈0.447`
  - `eval_reward≈-1.29`
  - `eval_success_rate=0.0`
- resumed controlled PPO under:
  - `outputs/training/bc_ppo/20260528_phase3_coverage_controlled_resume/phase3_coverage_controlled_resume/`
  - resumed improving after an early dip, but still had `eval_success_rate=0.0`
- BC coverage probe underperformed the PPO controlled line:
  - final `coverage_ratio_mean≈0.406`
  - final `success_rate_mean=0.0`
  - final `collision_rate_mean≈0.211`

Reason:

- coverage is not “randomly failing”; PPO can learn meaningful partial coverage behavior
- however, heuristic BC alone does not currently beat the best PPO controlled checkpoint
- next coverage work should prioritize long-horizon PPO tuning over generic heuristic BC

## Theme V: goal_nav convergence confirmation and coverage credit-assignment branch

Implemented:

- confirmed the DAgger BC -> ultra-strict PPO chain all the way through final eval:
  - `outputs/training/bc_ppo/20260528_goalnav_dagger_finetune_ultra_strict/goalnav_dagger_finetune_ultra_strict/eval_metrics.csv`
- added a faster coverage long-horizon / credit-assignment PPO config:
  - `configs/policy/debug_ppo_coverage_creditassign.yaml`
- launched a resumed coverage run from the best phase-3 checkpoint:
  - `outputs/training/bc_ppo/20260528_phase4_coverage_creditassign/phase4_coverage_creditassign/`

Current best confirmed goal_nav result:

- final `return_mean=10.1225`
- final `normalized_score_mean=1.0496`
- final `success_rate_mean=0.8`
- final `collision_rate_mean=0.0275`
- final `goal_coverage_ratio_mean=0.9425`

Current coverage status:

- heuristic BC probe stays weaker than PPO:
  - `coverage_ratio_mean≈0.406`
  - `success_rate_mean=0.0`
  - `collision_rate_mean≈0.211`
- coverage credit-assignment PPO currently has early eval points:
  - update 20: `eval_reward≈-0.551`, `coverage_ratio≈0.450`, `collision≈0.0899`, `success=0.0`
  - update 40: `eval_reward≈-1.008`, `coverage_ratio≈0.449`, `collision≈0.0944`, `success=0.0`

Reason:

- goal_nav is now strong enough to be treated as the current canonical specialist PPO success path
- coverage still needs iterative tuning, but the credit-assignment branch is now the best active mainline to continue from

## Theme W: coverage expert-v2 dataset and BC warm-start branch

Implemented:

- added `scripts/debug_long/generate_coverage_expert_v2_dataset.py`
- added `configs/policy/debug_ppo_coverage_expertv2_finetune_strict.yaml`
- generated stronger coverage expert dataset:
  - `outputs/debug_long/20260528_coverage_expert_v2/coverage_expert_v2_N4_e40.npz`
- trained BC on that dataset:
  - `outputs/training/bc/20260528_081939/debug_bc_coverage_coverage_N4_multi_channel_field_plus_task_id/`
- launched strict PPO from the new BC checkpoint:
  - `outputs/training/bc_ppo/20260528_phase6_coverage_expertv2_finetune/phase6_coverage_expertv2_finetune/`

Current evidence:

- quick expert-v2 probe over 5 episodes beat the canonical heuristic:
  - heuristic: `coverage_ratio≈0.420`, `collision≈0.130`, `intrinsic≈0.389`
  - expert-v2: `coverage_ratio≈0.579`, `collision≈0.0865`, `intrinsic≈0.555`
- BC trained on expert-v2 data also beat both the canonical heuristic BC and the PPO plateau:
  - final `coverage_ratio_mean≈0.512`
  - final `collision_rate_mean≈0.150`
  - final `return_mean≈-0.685`
  - final `normalized_score_mean≈1.532`

Reason:

- coverage heuristic BC from the canonical heuristic was not strong enough
- a stronger spread-aware expert policy produced the first clear coverage BC warm-start that surpassed the previous PPO coverage-ratio plateau

## Theme X: coverage coordination-repulsion PPO branch

Implemented:

- added optional `coordination_repulsion_strength` support to `policies/cnn_deepsets_policy.py`
- threaded the new config field through `policies/__init__.py`
- added coverage experiment configs:
  - `configs/policy/debug_ppo_coverage_expertv2_finetune_ultra_strict.yaml`
  - `configs/policy/debug_ppo_coverage_expertv2_repulsion.yaml`
- added a minimal regression test:
  - `tests/test_variable_policies.py::test_cnn_deepsets_policy_supports_optional_coordination_repulsion`

Current evidence:

- the plain ultra-strict PPO fine-tune from the expert-v2 BC checkpoint still degraded to about `coverage_ratio≈0.486`
- enabling repulsion bias produced the first clear post-BC PPO jump:
  - `outputs/training/bc_ppo/20260528_phase8_coverage_repulsion/phase8_coverage_repulsion/`
  - update 20: `coverage_ratio≈0.578`
  - update 20: `collision_rate≈0.0015`
  - update 20: `eval_reward≈5.397`
  - success is still `0.0`, so this is not yet full task convergence

Reason:

- coverage PPO had repeatedly plateaued near the heuristic structure ceiling
- a small explicit anti-crowding bias appears to help the factorized per-agent actor preserve spatial spread instead of collapsing back into redundant overlap

## Theme Y: coverage repulsion branch follow-up with reward-focus resume

Implemented:

- added `configs/env/debug_coverage_reward_focus.yaml`
- resumed a repulsion PPO checkpoint under the reward-focus coverage env:
  - `outputs/training/bc_ppo/20260528_phase9_coverage_repulsion_rewardfocus/phase9_coverage_repulsion_rewardfocus/`

Current evidence:

- the plain repulsion PPO branch remains the strongest PPO coverage line so far:
  - peak `coverage_ratio≈0.664`
  - `collision≈0.001`
  - one eval point reached `success_rate=0.05`
- adding reward-focus on top of the repulsion checkpoint did not further increase the coverage ratio:
  - update 20 under phase 9: `coverage_ratio≈0.665`
  - update 20 under phase 9: `success=0.0`

Reason:

- coverage reward scaling alone is not the main remaining blocker once anti-crowding coordination is in place
- the remaining gap now looks more like “last-mile completion behavior” than generic reward sparsity

## Theme Z: coverage spatial-action-head specialist branch

Implemented:

- extended `CNNDeepSetsPolicy` with optional:
  - `use_spatial_action_head`
  - `spatial_action_strength`
- added `configs/policy/debug_bc_coverage_spatialhead.yaml`
- added `configs/policy/debug_ppo_coverage_spatialhead_ultra_strict.yaml`
- trained a stronger coverage BC warm-start with expert-v2 data and the new spatial action head:
  - `outputs/training/bc/20260528_092949/debug_bc_coverage_spatialhead_coverage_N4_multi_channel_field_plus_task_id/`
- launched PPO from that checkpoint:
  - `outputs/training/bc_ppo/20260528_phase10_coverage_spatialhead_ultra/phase10_coverage_spatialhead_ultra/`

Current evidence:

- spatial-head BC outperformed the prior coverage BC:
  - `coverage_ratio_mean≈0.661`
  - `success_rate_mean≈0.05`
  - `collision_rate_mean≈0.00094`
  - `return_mean≈8.659`
- spatial-head ultra-strict PPO then preserved and slightly improved that structure:
  - update 20: `success≈0.10`, `coverage_ratio≈0.667`, `collision≈0.000625`
  - update 40: `success≈0.10`, `coverage_ratio≈0.670`, `collision≈0.0003125`
  - update 60: `success≈0.10`, `coverage_ratio≈0.682`, `collision≈0.00025`
  - final eval:
    - `return_mean≈8.083`
    - `normalized_score_mean≈2.615`
    - `coverage_ratio_mean≈0.667`
    - `success_rate_mean=0.0`
    - `collision_rate_mean≈0.003`

Reason:

- coverage needed a stronger geometric inductive bias than plain per-agent decoder plus repulsion alone
- an explicit spatial action head helps the actor map each agent to a spatial target instead of only producing raw waypoint deltas

## Theme AA: coverage spatial-target suppression probe

Implemented:

- added non-parametric `spatial_target_suppression_strength` and `spatial_target_suppression_sigma` support inside the spatial action head
- added `configs/policy/debug_ppo_coverage_spatialhead_suppression.yaml`
- verified the new branch with `tests/test_variable_policies.py`

Current evidence:

- suppression branch did not beat the plain spatial-head PPO branch:
  - update 20: `success≈0.10`, `coverage_ratio≈0.655`, `collision≈0.00044`
  - update 40: `success≈0.05`, `coverage_ratio` and reward both slipped relative to the no-suppression branch

Reason:

- explicit soft suppression was a reasonable next guess for target diversity
- but in current form it underperforms the simpler spatial-head + repulsion branch, so it should not replace the current mainline

## Theme AB: interruption-safe debug artifact writing

Implemented:

- `scripts/debug_long/collect_success_policy_dataset.py`
- `scripts/debug_long/collect_dagger_dataset.py`
- `scripts/debug_long/generate_success_expert_dataset.py`
- `scripts/debug_long/generate_coverage_expert_v2_dataset.py`

now write datasets and JSON sidecars via temporary files plus atomic rename.

Reason:

- long-running debug jobs were being interrupted by connection drops or accidental thread closure
- partial `.npz` writes were leaving corrupted datasets such as `BadZipFile`
- atomic write semantics keep the final path either fully valid or absent

## Theme AC: risk_nav and formation specialist repair

Implemented:

- added risk-nav configs:
  - `configs/policy/debug_bc_risk_nav_success.yaml`
  - `configs/policy/debug_ppo_risk_nav_success_ultra_strict.yaml`
- added formation configs:
  - `configs/policy/debug_bc_formation_success.yaml`
  - `configs/policy/debug_ppo_formation_success_ultra_strict.yaml`
  - `configs/policy/debug_bc_formation_dagger.yaml`
- generated success-only datasets:
  - `outputs/debug_long/20260528_risk_nav_success/risk_nav_success_N4_stride2.npz`
  - `outputs/debug_long/20260528_formation_success/formation_success_N4_stride2.npz`
- generated formation success-policy dataset:
  - `outputs/debug_long/20260528_formation_success_policy/formation_success_from_phase14_u40_plus_fullheuristic.npz`

Current evidence:

- risk_nav:
  - BC on full heuristic-state dataset reached `success_rate_mean≈0.55`
  - ultra-strict PPO from that BC checkpoint stabilized around `success≈0.55~0.60`
  - final line:
    - `outputs/training/bc_ppo/20260528_phase13_risk_nav_ultra/phase13_risk_nav_ultra/`
    - final `success_rate_mean=0.6`
    - final `goal_coverage_ratio_mean≈0.829`
    - final `collision_rate_mean≈0.049`
- formation:
  - success-only BC remained below heuristic
  - full-state BC improved but still lagged heuristic
  - ultra-strict PPO from the full-state BC checkpoint reached the first stable heuristic-level line:
    - update 40/60/80 held `success≈0.4`
    - final eval fell back to `success=0.3`
  - formation DAgger BC and formation success-policy BC did not beat that PPO branch

Reason:

- risk_nav was close enough to goal_nav that the BC -> strict PPO recipe transferred well
- formation responds to the current slot-aware actor and PPO warm-start, but its best result is still a best-checkpoint result rather than a final stabilized result

## Theme AD: human-readable summary doc and formation success-policy PPO branch

Implemented:

- added human-readable summary doc:
  - `docs/specialist_training_repair_summary_zh.md`
- launched formation success-policy PPO branch:
  - `outputs/training/bc_ppo/20260528_phase16_formation_successpolicy_ultra/phase16_formation_successpolicy_ultra/`

Current evidence:

- the doc now records:
  - what changed
  - why each code/config change was made
  - which task is currently repaired vs partially repaired
  - why conservative PPO / BC / DAgger / structure bias were technically justified
- formation success-policy PPO first eval currently holds:
  - update 20: `success≈0.4`
  - `formation_error≈0.091`
  - `collision≈0.0045`

Reason:

- the user explicitly requested a human-facing written summary under `docs/`
- frequent connection drops mean partial-but-important experimental state should be written to memory before runs finish

## Theme AE: coverage success-trajectory reinforcement follow-up

Implemented:

- collected additional coverage success-policy datasets:
  - deterministic from `phase10` update 60
  - deterministic from `phase10` update 40
  - noisy (`action_noise_std=0.05`) from `phase10` update 60
- tested several follow-up branches:
  - `phase15_coverage_successbc_ultra`
  - `phase17_coverage_sectorbias_ultra`
  - `phase18_coverage_stronger_repulsion`
- added optional `sector_target_bias_strength` and optional `use_global_slot_head` / `global_slot_strength`
- trained `debug_bc_coverage_globalslot`

Current evidence:

- success-policy dataset collected from `phase10` policy remained sparse:
  - deterministic `u60`: only `2` successful episodes over `240` attempts
  - deterministic `u40`: `0` successful episodes over `240` attempts
  - noisy `u60 + noise 0.05`: still only `2` successful episodes over `240` attempts
- `phase15_coverage_successbc_ultra` underperformed the plain `phase10` branch
- `phase17_coverage_sectorbias_ultra` underperformed the plain spatial-head branch
- `debug_bc_coverage_globalslot` underperformed the best `spatialhead BC`

Reason:

- the current coverage PPO line can enter the success region, but the success basin is still narrow and brittle
- adding weak success-only data is not enough by itself
- current lightweight group-level approximations have not yet surpassed the simpler spatial-head PPO branch

## Theme AF: risk_nav repaired and formation pushed to heuristic-level PPO

Implemented:

- generated success-only heuristic dataset for `risk_nav`
- trained full-state BC and ultra-strict PPO:
  - `outputs/training/bc_ppo/20260528_phase13_risk_nav_ultra/phase13_risk_nav_ultra/`
- generated success-only and full-state datasets for `formation`
- trained formation BC / PPO / DAgger BC / success-policy BC branches

Current evidence:

- `risk_nav` final line is now stable and reusable:
  - final `success_rate_mean=0.6`
  - final `goal_coverage_ratio_mean≈0.829`
  - final `collision_rate_mean≈0.049`
- `formation` best PPO line remains:
  - `outputs/training/bc_ppo/20260528_phase14_formation_ultra/phase14_formation_ultra/`
  - stable mid-run `success≈0.4`
  - final `success≈0.3`
- later formation success-policy / DAgger BC branches did not exceed that line

Reason:

- `risk_nav` transferred well from the `goal_nav` repair recipe
- `formation` benefits from structure-aware actor heads, but still does not hold its best success level all the way to final eval

## Theme AG: optional global-slot actor branch with cross-task compatibility

Implemented:

- extended `CNNDeepSetsPolicy` with optional:
  - `use_global_slot_head`
  - `global_slot_strength`
- added a cross-task compatibility test:
  - `tests/test_variable_policies.py::test_cnn_deepsets_global_slot_head_is_compatible_across_tasks`
- verified the expanded policy test suite passes with `9 passed`

Reason:

- the user explicitly required that any new task-allocation architecture remain compatible with the old centralized policy path
- the new group-level allocation branch therefore stays optional, default-off, and is now minimally validated across `goal_nav`, `coverage`, `formation`, and `risk_nav`

## Theme AH: PPO best-eval checkpoint preservation

Implemented:

- `algorithms/ppo.py` now saves:
  - `checkpoints/checkpoint_best_eval.pt`
  - `best_eval_summary.json`
- `scripts/train_ppo.py` variable-`N` loop mirrors the same behavior

Selection rule:

- prefer higher `eval_success_rate`
- break ties with higher `eval_reward`

Reason:

- `coverage` and `formation` often hit their best success mid-run and then drift downward by final eval
- preserving the best eval checkpoint avoids losing the most useful specialist snapshot just because the final checkpoint is worse

## Theme AI: final eval now prefers best PPO checkpoint

Implemented:

- `scripts/train_ppo.py` now accepts `--final_eval_source best|last`
- default is `best`
- if `checkpoints/checkpoint_best_eval.pt` exists, final per-task evaluation loads that checkpoint before writing `eval_metrics.csv`
- final eval rows now include `final_eval_source`

Reason:

- coverage and formation repeatedly showed “mid-run best, final worse”
- selecting the best checkpoint for final reporting is a pragmatic engineering fix that turns an unstable tail into a usable specialist artifact without changing the training update rule itself

## Theme AJ: coverage success-only reinforcement deepening

Implemented:

- collected a stronger success-heavy coverage dataset from `phase10` update 80 over a better seed window:
  - `outputs/debug_long/20260529_coverage_success_policy/coverage_success_from_phase10_u80_seed5000_plus_expertv2.npz`
  - `successful_episodes=15`
- extracted / iterated success-only mixes:
  - `coverage_success_only_from_phase10_u80_seed5000.npz`
  - `coverage_success_from_successBC_e40.npz`
  - `coverage_success_from_successBC_e40_round2.npz`
  - `coverage_success_from_successBC_e40_round3.npz`
- trained successive success-heavy BC refinements:
  - `outputs/training/bc/20260529_015503/...`
  - `outputs/training/bc/20260529_021428/...`
  - `outputs/training/bc/20260529_051145/...`
  - `outputs/training/bc/20260529_155058/...`

Current evidence:

- the first success-heavy BC pushed coverage BC success from `0.05` to `0.10`
- the next strengthened success-heavy BC pushed it further to:
  - `return_mean≈9.697`
  - `normalized_score_mean≈2.814`
  - `success_rate_mean≈0.15`
  - `collision_rate_mean≈0.0`
- PPO launched from these stronger success-only BC checkpoints (`phase20`, `phase23`) still did not exceed the `phase21 best` coverage PPO line

Reason:

- coverage success behavior can be partially reinforced through data, but the gain saturates before PPO turns that into a clearly better final specialist
- this suggests the remaining bottleneck is no longer “missing success examples” alone

## Theme AK: true group-level coverage actor candidates

Implemented:

- added optional `use_global_spatial_slot_head` / `global_spatial_slot_strength`
- added optional `actor_mean_residual_weight`
- added configs and probes for:
  - `global-slot BC`
  - `global-slot PPO`
  - `sector-bias PPO`
  - `slot-dominant BC`
- expanded `tests/test_variable_policies.py` to cover:
  - global slot head
  - slot-dominant actor
  - global-slot compatibility across tasks

Current evidence:

- `global-slot BC` and `slot-dominant BC` did not beat the best `spatial-head BC`
- `phase26_coverage_gspatialslot_bestfinal` first eval is still only `success≈0.05`
- the best coverage line remains the simpler `spatial-head PPO` branch with best-final checkpoint selection

Reason:

- lightweight group-level extensions are now present and compatible, but they have not yet beaten the current best coverage mainline
- further progress on coverage likely needs a stronger, more explicit group-level coordination design rather than another small bias term

## Theme P: Sequential PPO task-combination queue launcher

Implemented:

- added `scripts/run_ppo_task_queue.sh` for sequential PPO training over editable task-combination rows
- each queue row has its own label, task list, agent counts, update budget, eval episodes, GIF recording cadence, GPU selection, and extra CLI args
- queue runs remain headless, use TensorBoard, preserve `outputs/training/ppo/<timestamp>/<run_name>/`, and write queue logs under `outputs/training/task_queue/<timestamp>/`
- completion notification targets `muadib@foxmail.com` by default and supports SMTP env vars or local `mail` / `mailx` / `sendmail`
- `NOTIFY_ONLY=1` can now be used to test the notification path without starting training
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

## Theme R: Threaded environment batch backend

Implemented:

- added `ThreadEnvBatch` beside `SyncEnvBatch` in `utils/vector_env.py`, using `ThreadPoolExecutor` while preserving the same `envs`, `num_envs`, `reset(...)`, and `step(actions)` interface
- kept `SyncEnvBatch` behavior unchanged as the serial debug baseline
- extended `make_env_batch(...)` with `backend="sync|thread"` and optional `max_workers`
- added `make_task_balanced_env_batch(...)` to create a fixed number of environments per task with deterministic per-env seeds
- exported `ThreadEnvBatch` and `make_task_balanced_env_batch` through `utils/__init__.py`
- added `--env_backend`, `--envs_per_task`, and `--env_workers` to PPO, SAC, and TD3 training entrypoints
- added tests for threaded reset/step shape, ordered result collection, done-time auto reset, task-balanced env counts, forced task assignment, and task-balanced thread stepping

Reason:

- rollout collection was previously serialized over environment instances, which limits throughput when environment work dominates policy inference
- task-balanced batches make multi-task rollout sampling explicit by creating equal fixed-task environment groups instead of relying only on stochastic task sampling
- default backend remains `sync`, so all existing training commands preserve their old behavior unless a caller opts into `thread`

## Theme S: PPO multi-task suite launcher with specialist policies

Implemented:

- added `scripts/run_ppo_multitask_suite.sh` as a dedicated PPO experiment suite launcher
- the suite explicitly trains four independent specialist policies: `goal_nav`, `coverage`, `formation`, and `risk_nav`
- the suite also trains selected multi-task policies for pairs, triples, and all four tasks
- important tunables are listed at the top of the script, including config paths, scaling mode, observation variant, total updates, eval episodes, GIF cadence, env backend, task-balanced env count, thread workers, GPU id, and email settings
- each queue row carries its own task mix and training/evaluation/backend/GPU parameters
- queue logs and summary CSVs are written under `outputs/training/ppo_multitask_suite/<timestamp>/`, while training artifacts preserve the standard `outputs/training/ppo/<timestamp>/<run_name>/` layout
- `NOTIFY_ONLY=1` can now be used to test the suite notification path without running the queue

Reason:

- comparing specialist single-task policies with multi-task policies requires running both families under one repeatable queue
- keeping all important parameters visible in the script makes server-side experiment editing practical without changing trainer code

## Theme T: Shorter PPO run directory names

Implemented:

- shortened `scripts/train_ppo.py` run names from `<config>_<full_task_names>_N<agents>_<obs_variant>` to `<config>_<compact_task_tag>_N<agents>`
- compact task tags use `goal`, `cov`, `form`, `risk`, and `all4` for the canonical four-task set
- removed observation variant from the PPO run directory name because it is already saved in `snapshot/cli_args.yaml`

Reason:

- long PPO run directory names were cumbersome on server runs and nested output paths
- the removed fields are still recoverable from run snapshots and CSV metadata, so shortening the directory name does not remove experiment provenance

## Theme U: Queue-level PPO output grouping

Implemented:

- added optional `--run_timestamp` and `--run_name` arguments to `scripts/train_ppo.py`
- `scripts/run_ppo_task_queue.sh` and `scripts/run_ppo_multitask_suite.sh` now pass their launch timestamp to every child PPO run
- queue row labels are passed as child PPO run names, producing output paths such as `outputs/training/ppo/<queue_timestamp>/<run_label>/...`

Reason:

- a scripted training suite should group all child runs under the script launch time instead of scattering them across many per-run timestamps
- queue labels are clearer than long auto-generated names when comparing many independent runs from one launch

## Theme V: QQ SMTP notification configuration

Implemented:

- queue scripts now support a QQ SMTP preset with defaults for `smtp.qq.com:465` over SSL
- sensitive mail values are loaded from environment variables or an ignored local file at `.secrets/wayffusion_mail.env`
- added `configs/examples/wayffusion_mail.env.example` as the safe template for local SMTP configuration
- `.gitignore` now excludes `.secrets/` and `wayffusion_mail.env`
- notification self-test remains available with `NOTIFY_ONLY=1`

Reason:

- email authorization codes must not be committed into scripts
- QQ/Foxmail SMTP requires an authorization code, so scripts now explicitly warn when `SMTP_USER` or `SMTP_PASSWORD` is missing

Follow-up validation:

- fixed SMTP variables loaded from `.secrets/wayffusion_mail.env` so they are explicitly passed into the Python mail sender subprocess
- verified `NOTIFY_ONLY=1` succeeds for both `scripts/run_ppo_multitask_suite.sh` and `scripts/run_ppo_task_queue.sh`

## Theme W: Queue script GPU selection inheritance

Implemented:

- updated `scripts/run_ppo_multitask_suite.sh` so `DEFAULT_CUDA_VISIBLE_DEVICES` inherits an externally supplied `CUDA_VISIBLE_DEVICES`
- changed the default suite rows to leave their per-row GPU field empty, allowing commands such as `CUDA_VISIBLE_DEVICES=5 bash scripts/run_ppo_multitask_suite.sh` to run on physical GPU 5
- kept per-row GPU override support intact; filling the row GPU field still takes precedence over the inherited default

Reason:

- hard-coded per-row `cuda_visible_devices=0` caused external GPU selection to be ignored
- server users need a simple launch-time way to choose a GPU without editing every queue row

## Theme X: Goal-nav reward alignment repair

Implemented:

- `tasks/goal_nav.py` and `tasks/risk_nav.py` now rebuild `goal_reward` from unreached goals only
- `goal_progress` and distance shaping now use remaining goals instead of all goals
- `tests/test_rewards_basic.py` now guards the remaining-goal shaping contract

Reason:

- the previous static field and all-goal shaping combination could keep already reached goals attractive and distort progress rewards
- this change aligns reward shaping with the intended completion objective without lowering the success threshold

## Theme Y: Goal-nav debug-long diagnostics and DAgger workflow

Implemented:

- added `outputs/debug_long/20260528_035433/` experiment records for takeover, PPO baselines, BC, diagnostics, and summaries
- added `scripts/debug_long/analyze_ppo_run.py` to summarize PPO CSVs
- added `scripts/debug_long/diagnose_goal_nav_policy.py` to measure deterministic action alignment against remaining goal peaks
- added `scripts/debug_long/generate_success_expert_dataset.py` to build a success-only goal_nav BC dataset
- added `scripts/debug_long/collect_dagger_dataset.py` to collect learner states and relabel them with heuristic expert actions
- added `configs/policy/debug_bc_goal_nav_success.yaml` for the success-only BC run

Reason:

- PPO from random initialization produced zero success and a near-zero deterministic mean action
- successful-trajectory BC improved action direction but still had brittle collisions, so DAgger-style learner-state relabeling became the next repair path

## Theme Z: BC warm-start and PPO fine-tune hooks for goal_nav

Implemented:

- `scripts/train_bc.py` now accepts `--init_checkpoint`
- `algorithms/ppo.py` now clamps `log_std` immediately after checkpoint load and at trainer init
- `configs/policy/debug_ppo_goal_nav_bc_finetune_strict.yaml` is used for conservative BC-to-PPO fine-tuning

Reason:

- BC and PPO now share a reproducible warm-start path
- keeping the exploration scale fixed during fine-tune avoids reintroducing the BC checkpoint's looser action variance

## Theme AF: coverage completion-focused reward diagnostic

Implemented on 2026-05-30:

- `tasks/coverage.py` now supports two default-off reward components:
  - `coverage_shortfall`: per-step shaping based on the gap between current `coverage_ratio` and the unchanged success threshold
  - `failure_penalty`: one terminal penalty when the episode reaches `max_steps` without coverage success
- the default/base coverage task behavior is unchanged unless those weights are present in the env config
- added `configs/env/debug_coverage_completion_focus.yaml`
- added `configs/policy/debug_ppo_coverage_perm_ref_completion.yaml`
- added `tests/test_rewards_basic.py::test_coverage_failure_penalty_only_on_unsuccessful_timeout`

Reason:

- latest coverage runs show the policy often plateaus around `coverage_ratio≈0.70` with low collision but below the fixed `0.82` success threshold
- old reward settings paid large cumulative `coverage_level_reward` to non-successful long episodes, so PPO could prefer stable near-threshold failure over early successful termination
- the diagnostic branch keeps the success threshold unchanged, reduces the incentive to idle at partial coverage, raises the success terminal signal, and explicitly penalizes timed-out non-success

Validation:

- `/opt/conda/bin/python -m pytest -q tests/test_rewards_basic.py tests/test_bc_permutation_loss.py tests/test_variable_policies.py tests/test_ppo_episode_budget.py tests/test_expert_dataset.py`
- result: `22 passed`

Current coverage evidence before phase31:

- phase29 reference-regularized PPO best checkpoint:
  - `outputs/training/bc_ppo/20260530_phase29_coverage_permref/phase29_coverage_permref/checkpoints/checkpoint_best_eval.pt`
  - 50-episode eval: `success_rate_mean=0.16`, `coverage_ratio_mean≈0.702`, `collision_rate_mean≈0.00035`
- phase29 self-success BC dataset and BC replay did not improve:
  - dataset: `outputs/debug_long/20260530_coverage_expert_v3/coverage_success_round4_v3_phase29.npz`
  - 50-episode eval of self-success BC: `success_rate_mean=0.08`, `coverage_ratio_mean≈0.708`, `collision_rate_mean≈0.000875`
  - conclusion: simply cloning phase29 successful episodes regresses toward average partial-coverage behavior

Next experiment:

- phase31 should warm-start from phase29 best checkpoint and use the completion-focused env config
- this is a labeled reward-shaping diagnostic, not a lowered-threshold success claim

Phase31 result:

- run root: `outputs/training/bc_ppo/20260530_phase31_coverage_completion/phase31_coverage_completion/`
- best checkpoint update: `90`
- 20-episode final/best eval: `success_rate_mean=0.30`, `coverage_ratio_mean≈0.720`, `collision_rate_mean≈0.00025`
- 50-episode best eval: `success_rate_mean=0.16`, `coverage_ratio_mean≈0.707`, `collision_rate_mean≈0.000226`
- conclusion: completion-focused reward fixes the objective semantics and maintains the 20-episode peak, but does not improve 50-episode robustness over phase29

Additional coverage expert diagnostics:

- added `CoverageExpertV4` in `scripts/debug_long/generate_coverage_expert_v2_dataset.py`
  - stateful sector assignment
  - persistent per-agent targets
  - short-range collision repulsion
- extended `scripts/debug_long/generate_success_expert_dataset.py --teacher` with `coverage_expert_v4`
- validation:
  - `/opt/conda/bin/python -m pytest -q tests/test_expert_dataset.py tests/test_rewards_basic.py tests/test_bc_permutation_loss.py tests/test_variable_policies.py`
  - result: `21 passed`
- direct 80-episode V4 eval:
  - output: `outputs/debug_long/20260530_coverage_expert_v4/coverage_expert_v4_eval_e80.summary.json`
  - `success_rate=0.0`, `coverage_ratio_mean≈0.567`, `collision_rate_mean≈0.0161`
  - conclusion: V4 is worse than phase29 policy and should not be used for BC/PPO
- direct 80-episode band-sweep diagnostic:
  - output: `outputs/debug_long/20260530_coverage_band_sweep/band_sweep_eval_e80.summary.json`
  - `success_rate=0.0`, `coverage_ratio_mean≈0.596`, `collision_rate_mean≈0.0127`
  - conclusion: simple hand-coded lane/band sweep is not a viable teacher

Phase32 result:

- added `configs/policy/debug_ppo_coverage_perm_ref_completion_explore.yaml`
- run root: `outputs/training/bc_ppo/20260530_phase32_coverage_completion_explore/phase32_coverage_completion_explore/`
- warm-start: phase29 best checkpoint
- key change: higher exploration (`log_std≈-1.2`), lower reference regularization, 2 PPO epochs, small entropy bonus
- stopped manually after drift because best checkpoint was already saved:
  - best checkpoint update: `20`
  - 20-episode best eval: `success_rate_mean=0.35`
  - later eval drifted to `0.15` at update 50 and `0.10` at update 60
- 50-episode best eval:
  - `success_rate_mean=0.20`
  - `coverage_ratio_mean≈0.703`
  - `collision_rate_mean≈0.000125`
- conclusion: exploration can find slightly better coverage specialists than phase29/31, but the improvement is still weak and drifts without stronger stabilization

Phase33 result:

- added `configs/policy/debug_ppo_coverage_perm_ref_completion_stabilize.yaml`
- run root: `outputs/training/bc_ppo/20260530_phase33_coverage_completion_stabilize/phase33_coverage_completion_stabilize/`
- warm-start: phase32 best checkpoint
- key change: lower LR (`3e-6`), stronger reference regularization (`0.35`), lower fixed exploration than phase32
- training behavior:
  - update20 eval `success_rate=0.35`
  - update40 eval `success_rate=0.35`
  - no phase32-style collapse through update80
- 50-episode h200 best eval:
  - `success_rate_mean=0.20`
  - `coverage_ratio_mean≈0.707`
  - `collision_rate_mean≈0.000125`
- conclusion: phase33 stabilizes the 20-episode `0.35` peak but still only generalizes to `0.20` on h200

Horizon feasibility diagnostic:

- added `configs/env/debug_coverage_completion_focus_h300.yaml`
- this config keeps `coverage.success_ratio=0.82` unchanged but increases `max_steps` from 200 to 300
- same phase33 checkpoint evaluated under h300:
  - output: `outputs/training/bc_ppo/20260530_phase33_coverage_completion_stabilize/phase33_coverage_completion_stabilize/eval_best_50_h300_diagnostic/`
  - `success_rate_mean=0.48`
  - `coverage_ratio_mean≈0.769`
  - `collision_rate_mean≈0.000198`
- interpretation:
  - the h200 policy already knows a useful coverage behavior
  - many h200 failures are time/path-budget failures, because allowing more steps without lowering the threshold nearly doubles the 50-episode success rate
  - h300 is a diagnostic environment change and must not be mixed with h200 canonical results

Phase34 h300 PPO result:

- run root: `outputs/training/bc_ppo/20260530_phase34_coverage_h300_stabilize/phase34_coverage_h300_stabilize/`
- warm-start: phase33 best checkpoint
- env: `configs/env/debug_coverage_completion_focus_h300.yaml`
- best checkpoint update: `60`
- 20-episode best/final-source eval:
  - `success_rate_mean=0.65`
  - `collision_rate_mean=0.0`
- 50-episode best eval:
  - `success_rate_mean=0.56`
  - `coverage_ratio_mean≈0.770`
  - `collision_rate_mean≈0.000135`
  - `path_length_mean≈1.410`
- conclusion:
  - coverage PPO can train a substantially stronger expert when the horizon allows enough path budget
  - current h200 coverage remains unresolved; current h300 coverage is the strongest diagnostic specialist

## Theme AG: factorized group actor policy

Implemented on 2026-05-30:

- added `policies/factorized_group_policy.py`
- added `policy_class: factorized_group` to `policies/__init__.py`
- added configs:
  - `configs/policy/ppo_factorized_group.yaml`
  - `configs/policy/debug_bc_goal_nav_factorized_group.yaml`
  - `configs/policy/debug_ppo_goal_nav_factorized_group_finetune.yaml`
  - `configs/policy/debug_bc_coverage_factorized_group_perm.yaml`
  - `configs/policy/debug_ppo_coverage_factorized_group_h300.yaml`
- added tests in `tests/test_variable_policies.py`

Architecture contract:

- critic remains centralized and still consumes one global state feature
- actor remains shared across UAVs and still outputs `[B, N, 2]`
- a small bank of learned group tokens is conditioned on the global state
- each agent soft-assigns to those group tokens
- group tokens optionally select spatial slots from the field and feed both:
  - a per-agent group context into the actor decoder
  - a group-coordinate waypoint bias
- `agent_mask` and variable `N` stay supported
- PPO joint log-prob aggregation remains unchanged because the action head still emits one factorized `[N,2]` action tensor per env

Reason:

- the old `CNNDeepSetsPolicy` already had a shared per-agent decoder, but all agents received the same pooled swarm context
- this new policy inserts an explicit coordination layer between pooled state and per-agent action decoding, matching the requested “single-agent / small-group waypoint allocation under centralized information sharing” direction

Validation:

- `/opt/conda/bin/python -m pytest -q tests/test_variable_policies.py tests/test_rewards_basic.py tests/test_bc_permutation_loss.py tests/test_expert_dataset.py`
- result after policy integration: `23 passed`
- final recheck after all edits:
  - `/opt/conda/bin/python -m pytest -q tests/test_rewards_basic.py tests/test_expert_dataset.py tests/test_bc_permutation_loss.py tests/test_variable_policies.py tests/test_ppo_episode_budget.py`
  - result: `22 passed`

Goal-nav factorized-group specialist:

- BC run:
  - `outputs/training/bc/20260530_082030/debug_bc_goal_nav_factorized_group_goal_nav_N4_multi_channel_field_plus_task_id/`
  - dataset: `outputs/debug_long/20260528_035433/datasets/goal_nav_dagger_from_bcppo060_plus_success.npz`
  - 20-episode eval: `success_rate_mean=0.75`, `goal_coverage_ratio_mean≈0.855`
- PPO fine-tune:
  - `outputs/training/bc_ppo/20260530_phase36_goalnav_factorized_group_ppo/phase36_goalnav_factorized_group_ppo/`
  - 20-episode best/final-source eval: `success_rate_mean=0.80`, `goal_coverage_ratio_mean≈0.872`
  - 50-episode best eval:
    - `success_rate_mean=0.78`
    - `goal_coverage_ratio_mean≈0.877`
    - `collision_rate_mean≈0.056`
- conclusion:
  - the new factorized-group policy can train a real PPO specialist expert on `goal_nav`
  - this de-risks the new decision architecture itself

Coverage factorized-group specialist:

- BC run:
  - `outputs/training/bc/20260530_083743/debug_bc_coverage_factorized_group_perm_coverage_N4_multi_channel_field_plus_task_id/`
  - dataset: `outputs/debug_long/20260530_coverage_expert_v3/coverage_success_round4_plus_v3_e40.npz`
  - 20-episode eval: `success_rate_mean=0.20`, `coverage_ratio_mean≈0.700+`, `collision_rate_mean=0.0`
- h300 PPO fine-tune:
  - `outputs/training/bc_ppo/20260530_phase37_coverage_factorized_group_h300/phase37_coverage_factorized_group_h300/`
  - 20-episode final-source eval: `success_rate_mean=0.50`, `coverage_ratio_mean≈0.768`, `collision_rate_mean≈0.0`
  - 50-episode best eval:
    - `success_rate_mean=0.40`
    - `coverage_ratio_mean≈0.768`
    - `collision_rate_mean≈0.00138`
- comparison against old h300 specialist:
  - old phase34 h300 50-episode eval: `success_rate_mean=0.56`, `coverage_ratio_mean≈0.770`, `collision_rate_mean≈0.000135`
- conclusion:
  - the new architecture is viable for coverage but is not yet the best-performing coverage policy
  - current factorized-group coverage issue is not “cannot train at all”; it is “path efficiency / stability still worse than old spatial-head line”
