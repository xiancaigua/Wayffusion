# 建议的下一步工作

## 高优先级

### 1. 转移主线到 coverage specialist 修复

建议：

- 继续 coverage 的 expert-v2 BC -> spatial-head PPO 主线
- 优先观察 `coverage_ratio` 是否能稳定跨过 `0.5`、`0.6`、再接近 `0.82`
- 当前 best-final coverage PPO 已经把 specialist 推到更高一点的成功区：
  - best eval `success≈0.15`
  - `coverage_ratio≈0.673`
  - `collision≈0.00019`
  - run root:
    `outputs/training/bc_ppo/20260529_phase24_coverage_successonly_v2_bestfinal/phase24_coverage_successonly_v2_bestfinal/`
- 当前更具体的 coverage 主线是：
  - `expert-v2 dataset -> spatial-head BC -> spatial-head PPO`
  - 如果继续推进，优先用成功策略轨迹继续做 success-heavy BC / DAgger，而不是继续单纯加 reward scale
  - 这条成功轨迹强化链已经做过一轮：success-heavy BC、sector-bias、stronger-repulsion 都没有超过 `phase10`
  - 因此下一步应升级到真正的 group-level coordination actor，而不是继续做轻量偏置微调
- canonical heuristic BC 只保留为结构探针，不作为 coverage 主线

原因：

- `goal_nav` 当前已经有一条可复用 specialist 收敛链，并且 PPO final eval 已稳定在 `success_rate_mean=0.8`
- `coverage` 现在才是剩余的主要阻塞项
- 可选 `global slot head` 已经接入代码并通过跨任务兼容性测试，因此真正的 group-level actor 分支现在具备可实验条件

### 2. 刷新正式 verification 文件

建议：

- 重新生成一个和当前目录结构、测试数量、命令入口一致的 `outputs/verification.md`

原因：

- 当前文件已明显陈旧，会误导后续 agent 和用户

### 3. 统一配置命名

建议：

- 在 `ppo_mlp / mlp_ppo`
- `ppo_cnn_deepsets / cnn_deepsets_ppo`

两套命名中选一套作为主路径，并把另一套标记为 legacy 或 alias。

原因：

- 当前重复命名已经进入维护负担区间

### 4. 为评估脚本补 `--latest`

建议：

- 给 `evaluate_policy.py` 和 `evaluate_scaling.py` 增加最新 run 自动发现

原因：

- 训练目录现在有时间层，手工填 checkpoint 路径更容易出错

## 中优先级

### 5. 在 coverage 主线稳定后，再刷新 canonical specialist / multitask runs

建议：

- 用当前新的 `checkpoints/` 与 `snapshot/` 规范，跑一条标准多任务 PPO 训练线
- 同时保留 eval media 和 final eval artifacts

原因：

- 这样后续所有 agent 都能引用“新结构下的标准 run”

### 6. 为 agent_memory 增加自动生成脚本

建议：

- 后续可考虑增加脚本，把当前测试数、最新 run、最新 checkpoint 自动写回 manifest 或某张卡

原因：

- 当前 memory cards 是人工审计产物，后面若频繁修改，自动刷新会更稳

## 低优先级

### 7. 处理 `sitecustomize.py` 重复

建议：

- 明确是否保留双副本

### 8. 历史 outputs 分层归档

建议：

- 把旧结构 run 和新结构 run 分层标记

原因：

- 未来自动扫 `outputs/` 做比较时会更干净

## 路由建议

如果后续 agent 需要继续主线开发，建议优先路线是：

1. Keep `goal_nav` factorized-group as repaired:
   `outputs/training/bc_ppo/20260530_phase36_goalnav_factorized_group_ppo/phase36_goalnav_factorized_group_ppo/checkpoints/checkpoint_best_eval.pt`.
2. Keep `formation` factorized-group as provisionally repaired:
   `outputs/training/bc_ppo/20260530_phase39_formation_factorized_group_ppo/phase39_formation_factorized_group_ppo/checkpoints/checkpoint_best_eval.pt`.
3. Risk-nav under factorized-group is still not repaired:
   current best 50-episode eval is `success_rate_mean=0.24`, with collision `≈0.177`.
4. Next risk-nav route should not be simple repulsion/reference or low-collision-only BC; both failed.
5. For risk-nav, try a richer DAgger dataset from learner states relabeled by the risk heuristic, or add a task-specific risk/goal spatial head while keeping `policy_class: factorized_group`.
6. Coverage remains unresolved under h200 and weaker than old h300 under new architecture:
   new factorized-group h300 best is `success_rate_mean=0.40`; old spatial-head h300 best is `0.56`.
7. For coverage, next new-architecture route is group-specific coverage suppression or persistent group targets, not more heuristic V4/band-sweep data.
8. Refresh `outputs/verification.md`, checkpoint latest helpers, and canonical multitask runs after risk_nav and coverage have new-architecture specialists.

## 当前 continuation 路线：只推进 factorized_group

约束：

- 不再调旧 `CNNDeepSetsPolicy` 主线。
- 不改环境动力学、任务定义、成功阈值或核心任务要素。
- 允许 reward-only 配置实验，但必须显式记录为 debug/training config。

立即执行：

1. `risk_nav`: 用 `scripts/debug_long/collect_dagger_dataset.py` 从 factorized-group learner rollout 采样状态，并用 heuristic teacher relabel，输出到 `outputs/debug_long/20260530_factorized_group_continuation/`。
2. `risk_nav`: 用原 success dataset + DAgger dataset 训练 factorized-group BC，再用 `configs/policy/debug_ppo_risk_nav_factorized_group_dagger_safe.yaml` 和 `configs/env/debug_risk_nav_safety_completion.yaml` 做保守 PPO。
3. `coverage`: 从 factorized-group h300 best checkpoint 继续 completion-focused PPO，判断剩余问题是 episode budget/credit assignment 还是架构表达不足。
4. 每个新增数据集、训练 run、代码或配置修改都必须同步写回 `agent_memory/cards/03_recent_modifications.md` 和本卡。

Risk-nav continuation update:

- DAgger BC improved factorized-group risk-nav to 30-episode `success_rate_mean=0.533`, but collision remains high at `0.114`.
- Next action is conservative PPO from `outputs/training/bc/20260530_121227/debug_bc_risk_nav_factorized_group_risk_nav_N4_multi_channel_field_plus_task_id/checkpoints/checkpoint_0040.pt` using `configs/policy/debug_ppo_risk_nav_factorized_group_dagger_safe.yaml` and `configs/env/debug_risk_nav_safety_completion.yaml`.

Risk-nav status after phase41:

- Treat `risk_nav` as provisionally repaired under `factorized_group`.
- Current best checkpoint: `outputs/training/bc_ppo/20260530_phase41_risknav_factorized_group_dagger_safe/phase41_risknav_factorized_group_dagger_safe/checkpoints/checkpoint_best_eval.pt`.
- Evidence: independent 100-episode eval `success_rate_mean=0.65`, `collision_rate_mean=0.0208`.
- Remaining validation: run at least one seed repeat after coverage is stabilized.

Coverage next:

- Do not use h300 as the canonical success claim because the user disallowed environment changes except reward.
- Continue with h200 and reward-only completion shaping first.

Coverage h200 experiment to run:

- Start from factorized-group coverage BC checkpoint `outputs/training/bc/20260530_083743/debug_bc_coverage_factorized_group_perm_coverage_N4_multi_channel_field_plus_task_id/checkpoints/checkpoint_0024.pt`.
- Train with `configs/policy/debug_ppo_coverage_factorized_group_completion_h200.yaml` and `configs/env/debug_coverage_completion_focus.yaml`.
- Evaluate with at least 30 episodes during training and 100 episodes for the best checkpoint before claiming repair.

Coverage phase42 result:

- Direct utility-slot PPO failed and was stopped early.
- Do not continue phase42.
- Next action: train BC with the utility-enabled factorized-group policy on the existing coverage dataset, then PPO fine-tune from that BC checkpoint.

Coverage utility BC result:

- Utility-head BC failed despite low supervised loss: 30-episode success `0.0`.
- Do not PPO from `outputs/training/bc/20260530_135952/...`.
- Next action: coverage learner-state DAgger using phase37 factorized-group coverage learner as rollout policy and `coverage_expert_v2` as teacher, then BC/PPO with the original non-utility factorized-group config.

Coverage teacher diagnostic result:

- Existing coverage teachers are not h200-successful; do not expect expert-v2/v3/v4 DAgger alone to solve coverage.
- Next action: run `configs/policy/debug_ppo_coverage_factorized_group_h200_stable.yaml` from the original factorized-group coverage BC checkpoint under `configs/env/debug_coverage_completion_focus.yaml`.

Coverage phase43 result:

- Stable h200 PPO from original BC did not exceed the original `0.20` success baseline.
- Current best canonical h200 factorized-group coverage remains around `success_rate_mean=0.20`.
- Next route should be reward-only coverage curriculum/shaping or a newly designed high-success h200 coverage teacher; do not repeat expert-v2/v3/v4 DAgger or utility-head PPO as-is.

Coverage ratio-curriculum next:

- Run `configs/policy/debug_ppo_coverage_factorized_group_h200_ratio_curriculum.yaml` from the original factorized-group BC checkpoint under `configs/env/debug_coverage_ratio_curriculum.yaml`.
- If it improves success above `0.20`, do independent 100-episode eval. If not, coverage remains blocked on high-success h200 expert/curriculum design.

Coverage phase44 result:

- Ratio-curriculum reward-only PPO failed to improve beyond `0.20` baseline; best was `0.1667`.
- If using lower-threshold curriculum next, mark it explicitly as training-only/diagnostic and validate final checkpoint under canonical `coverage.success_ratio=0.82`, h200.

Coverage milestone PPO next:

- Run `configs/policy/debug_ppo_coverage_factorized_group_h200_milestone.yaml` with `configs/env/debug_coverage_milestone_reward.yaml` from the original factorized-group BC checkpoint.
- If best 30-episode success exceeds `0.20`, run 100-episode canonical eval. If it fails, coverage remains blocked on stronger curriculum/high-success h200 expert design.

Coverage phase45 canonical validation:

- Milestone PPO is not a canonical repair. Canonical 100-episode success was `0.17` with repeated coverage ratio `0.991`.
- Next credible route: implement an anti-revisit / frontier-sweep training signal or a high-success h200 teacher, then validate under original `configs/env/coverage.yaml`.

Coverage anti-revisit PPO next:

- Run `configs/policy/debug_ppo_coverage_factorized_group_h200_antirevisit.yaml` with `configs/env/debug_coverage_antirevisit_reward.yaml` from the original factorized-group BC checkpoint.
- Canonical validation remains mandatory under `configs/env/coverage.yaml`.

Coverage phase46 canonical validation:

- Anti-revisit PPO is not a canonical repair. Canonical 100-episode success is `0.20`, repeated coverage remains `0.99125`.
- Stop repeating scalar reward-only PPO from the same BC checkpoint.
- Next credible implementation: add a factorized-group-compatible persistent target/frontier assignment mechanism or create a true high-success h200 coverage teacher, then validate under canonical `configs/env/coverage.yaml`.

Coverage success-heavy BC result:

- Success-only BC from phase46 successful trajectories failed: 50-episode canonical success `0.140`.
- Stop using current success-only data as the main coverage fix.
- Next implementation should be architectural but compatible with `factorized_group`: persistent per-agent/group frontier target assignment, or alternatively build a true h200 coverage teacher before BC/PPO.

Post-continuation status:

- Regression tests passed: `26 passed`.
- Do not claim coverage repaired. Current canonical h200 coverage is still around `0.20` success.
- Best next coverage route is not another scalar reward tweak; implement persistent per-agent/group frontier targets in the new architecture, or first build a true high-success h200 coverage teacher.

Coverage frontier-slot next:

- Train `configs/policy/debug_bc_coverage_factorized_group_frontier_perm.yaml` on `outputs/debug_long/20260530_factorized_group_continuation/coverage_success_from_phase46_canonical_plus_v3.npz`.
- If BC canonical eval is not worse than baseline, run `configs/policy/debug_ppo_coverage_factorized_group_frontier.yaml` under canonical or anti-revisit reward and validate with canonical 100 episodes.

Coverage frontier-low next:

- Run `configs/policy/debug_ppo_coverage_factorized_group_frontier_low.yaml` from the original coverage BC checkpoint under anti-revisit reward.
- If it does not exceed `0.20`, frontier bias as currently implemented should be considered ineffective and the next route should be a better h200 teacher rather than more frontier tuning.

Coverage frontier-low result:

- Phase48 failed and was stopped early. Low-strength frontier bias did not preserve baseline behavior.
- The next coverage path should be a high-success h200 teacher/trajectory generator, not additional frontier-bias tuning.

Coverage teacher prototype result:

- Local greedy and waypoint-lookahead h200 teacher prototypes failed.
- If continuing coverage, build an explicit route-planning/sweep teacher or add recurrent/persistent target memory with proper state handling; do not rely on local greedy scoring.

Frontier continuation verification:

- Regression tests passed: `28 passed`.
- No active training/evaluation process is running.
- Coverage is still unresolved; current implemented frontier bias is compatible but empirically ineffective.

Documentation update:

- Current explanation of pure PPO unreliability and BC/DAgger necessity is in `docs/specialist_ppo_reliability_zh.md`.
- Future agents should read that doc before proposing another pure-PPO or scalar reward-only coverage run.
