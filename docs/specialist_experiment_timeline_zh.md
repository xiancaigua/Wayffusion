# Specialist 实验时间线与输出目录说明

本文专门回答两个问题：

1. `outputs/` 里这些目录分别是什么、对应哪条实验主线？
2. 从最开始到现在，我到底按什么顺序做了哪些实验，目的是什么，结果怎么样？

## 1. 当前推荐直接看的结果目录

### goal_nav

当前 canonical run：

- [outputs/training/bc_ppo/20260528_goalnav_dagger_finetune_ultra_strict/goalnav_dagger_finetune_ultra_strict](/workspace/Wayffusion/Wayffusion/outputs/training/bc_ppo/20260528_goalnav_dagger_finetune_ultra_strict/goalnav_dagger_finetune_ultra_strict)

当前结论：

- 已修通
- final `success_rate_mean=0.8`

### coverage

当前最好可复用 PPO run：

- [outputs/training/bc_ppo/20260528_phase21_coverage_bestfinal/phase21_coverage_bestfinal](/workspace/Wayffusion/Wayffusion/outputs/training/bc_ppo/20260528_phase21_coverage_bestfinal/phase21_coverage_bestfinal)

说明：

- 这是在 `phase10 spatial-head ultra-strict PPO` 基础上，重新跑并启用了 “best eval checkpoint 做 final eval”
- `best_eval_summary.json` 记录了当前最优 checkpoint
- 当前 best checkpoint 指向：
  - `update=60`
  - `eval_success_rate=0.1`
  - `eval_reward=8.4501`

当前最好 BC warm-start：

- [outputs/training/bc/20260529_021428/debug_bc_coverage_successonly_coverage_N4_multi_channel_field_plus_task_id](/workspace/Wayffusion/Wayffusion/outputs/training/bc/20260529_021428/debug_bc_coverage_successonly_coverage_N4_multi_channel_field_plus_task_id)

说明：

- 这是 success-heavy coverage BC 的第二版
- final `success_rate_mean=0.15`

### risk_nav

当前 canonical run：

- [outputs/training/bc_ppo/20260528_phase13_risk_nav_ultra/phase13_risk_nav_ultra](/workspace/Wayffusion/Wayffusion/outputs/training/bc_ppo/20260528_phase13_risk_nav_ultra/phase13_risk_nav_ultra)

当前结论：

- 已修通
- final `success_rate_mean=0.6`

### formation

当前最好 PPO run：

- [outputs/training/bc_ppo/20260528_phase14_formation_ultra/phase14_formation_ultra](/workspace/Wayffusion/Wayffusion/outputs/training/bc_ppo/20260528_phase14_formation_ultra/phase14_formation_ultra)

当前最好 BC run：

- [outputs/training/bc/20260529_025330/debug_bc_formation_dagger_formation_N4_multi_channel_field_plus_task_id](/workspace/Wayffusion/Wayffusion/outputs/training/bc/20260529_025330/debug_bc_formation_dagger_formation_N4_multi_channel_field_plus_task_id)

当前结论：

- 接近修通
- 但还没有像 `goal_nav / risk_nav` 那样完全稳定

## 2. 从开始到现在的关键实验顺序

### 阶段 A：确认 baseline 问题是真的存在

1. `goal_nav baseline PPO`

目录：

- [outputs/training/ppo/20260527_132850_goal_nav_baseline/debug_goal_nav_baseline](/workspace/Wayffusion/Wayffusion/outputs/training/ppo/20260527_132850_goal_nav_baseline/debug_goal_nav_baseline)

目的：

- 看从零训练的 PPO 到底是不是完全不学

结果：

- final `success_rate_mean=0.0`
- `goal_coverage_ratio_mean≈0.207`

结论：

- 不是偶然没学到，而是当前设定下真的不收敛

2. `goal_nav controlled PPO`

目录：

- [outputs/training/ppo/20260527_132850_goal_nav_controlled/debug_goal_nav_controlled](/workspace/Wayffusion/Wayffusion/outputs/training/ppo/20260527_132850_goal_nav_controlled/debug_goal_nav_controlled)

目的：

- 验证更保守超参能否直接修通

结果：

- final `success_rate_mean=0.0`
- 但中期比 baseline 更安全

结论：

- 单纯调 PPO 超参不够

### 阶段 B：修 observability 与 PPO 诊断

代码修改：

- `goal_nav / risk_nav` 的 `goal_reward` 改成只显示未完成目标
- PPO 加了 `log_std clamp`、`approx_kl`、`clip_frac`、`grad_norm`、rollout 终端统计

目的：

- 排除“观测错误”和“诊断不足”

结论：

- 这一步必要，但仍不足以直接修通 `goal_nav`

### 阶段 C：goal_nav 成功轨迹 BC -> DAgger -> PPO

1. success-only BC

2. BC + conservative PPO

3. DAgger dataset

关键 debug 记录：

- [outputs/debug_long/20260528_035433](/workspace/Wayffusion/Wayffusion/outputs/debug_long/20260528_035433)

关键最终产物：

- [outputs/training/bc_ppo/20260528_goalnav_dagger_finetine_ultra_strict/goalnav_dagger_finetine_ultra_strict](/workspace/Wayffusion/Wayffusion/outputs/training/bc_ppo/20260528_goalnav_dagger_finetine_ultra_strict/goalnav_dagger_finetine_ultra_strict)

结果：

- `success_rate_mean=0.8`

结论：

- `goal_nav` 修通

### 阶段 D：coverage 从“部分覆盖”推进到“进入成功区”

1. controlled PPO

目录：

- `phase2_coverage_controlled`
- `phase3_coverage_controlled_resume`
- `phase4_coverage_creditassign`
- `phase5_coverage_rewardfocus`

目的：

- 先看只靠 PPO 和 conservative exploration 能走多远

结果：

- 最初把 coverage ratio 从随机水平推到了 `0.45` 左右
- 但 success 仍是 `0`

结论：

- 不是完全学不动
- 但需要更强结构偏置

2. `coverage expert v2`

目录：

- [outputs/debug_long/20260528_coverage_expert_v2](/workspace/Wayffusion/Wayffusion/outputs/debug_long/20260528_coverage_expert_v2)

目的：

- 构造比原 heuristic 更强的 coverage teacher

结果：

- 比原 heuristic 更高的 coverage ratio，更低 collision

3. `spatial-head BC`

目录：

- [outputs/training/bc/20260528_092949/debug_bc_coverage_spatialhead_coverage_N4_multi_channel_field_plus_task_id](/workspace/Wayffusion/Wayffusion/outputs/training/bc/20260528_092949/debug_bc_coverage_spatialhead_coverage_N4_multi_channel_field_plus_task_id)

目的：

- 把空间几何偏置直接灌进 actor

结果：

- `coverage_ratio_mean≈0.661`
- `success_rate_mean≈0.05`

4. `spatial-head PPO`

目录：

- [outputs/training/bc_ppo/20260528_phase10_coverage_spatialhead_ultra/phase10_coverage_spatialhead_ultra](/workspace/Wayffusion/Wayffusion/outputs/training/bc_ppo/20260528_phase10_coverage_spatialhead_ultra/phase10_coverage_spatialhead_ultra)

目的：

- 在强 BC 起点上，用 ultra-strict PPO 保住并扩大成功区

结果：

- 多个 eval 点稳定 `success≈0.10`
- `coverage_ratio≈0.67~0.68`
- `collision≈0.00025~0.00063`

结论：

- `coverage` 已经从不收敛推进到“连续进入成功区”
- 但 final 还会回落

5. success-heavy coverage BC

关键目录：

- [outputs/training/bc/20260529_015503/debug_bc_coverage_successonly_coverage_N4_multi_channel_field_plus_task_id](/workspace/Wayffusion/Wayffusion/outputs/training/bc/20260529_015503/debug_bc_coverage_successonly_coverage_N4_multi_channel_field_plus_task_id)
- [outputs/training/bc/20260529_021428/debug_bc_coverage_successonly_coverage_N4_multi_channel_field_plus_task_id](/workspace/Wayffusion/Wayffusion/outputs/training/bc/20260529_021428/debug_bc_coverage_successonly_coverage_N4_multi_channel_field_plus_task_id)
- [outputs/training/bc/20260529_051145/debug_bc_coverage_successonly_coverage_N4_multi_channel_field_plus_task_id](/workspace/Wayffusion/Wayffusion/outputs/training/bc/20260529_051145/debug_bc_coverage_successonly_coverage_N4_multi_channel_field_plus_task_id)

目的：

- 用成功尾段行为去补 coverage 的最后一推

结果：

- BC 成功率从 `0.05` 提高到 `0.15`

结论：

- success-heavy 数据是有效的
- 但后续 PPO 还没明显超过 `phase10/phase21`

6. best-final coverage PPO

目录：

- [outputs/training/bc_ppo/20260528_phase21_coverage_bestfinal/phase21_coverage_bestfinal](/workspace/Wayffusion/Wayffusion/outputs/training/bc_ppo/20260528_phase21_coverage_bestfinal/phase21_coverage_bestfinal)

目的：

- 把中段最优 checkpoint 固化成正式最终产物，而不是被 final 回落污染

结果：

- final eval now uses `checkpoint_best_eval.pt`
- current best preserved coverage result:
  - `success_rate_mean=0.1`
  - `coverage_ratio_mean≈0.669`
  - `collision_rate_mean≈0.000625`

### 阶段 E：risk_nav

路径：

- full-state heuristic dataset
- BC
- ultra-strict PPO

目录：

- [outputs/training/bc_ppo/20260528_phase13_risk_nav_ultra/phase13_risk_nav_ultra](/workspace/Wayffusion/Wayffusion/outputs/training/bc_ppo/20260528_phase13_risk_nav_ultra/phase13_risk_nav_ultra)

结果：

- `success_rate_mean=0.6`

结论：

- `risk_nav` 修通

### 阶段 F：formation

路径：

- full-state heuristic BC
- ultra-strict PPO
- DAgger / success-policy BC 增强

当前最好目录：

- [outputs/training/bc_ppo/20260528_phase14_formation_ultra/phase14_formation_ultra](/workspace/Wayffusion/Wayffusion/outputs/training/bc_ppo/20260528_phase14_formation_ultra/phase14_formation_ultra)

结果：

- 中段能稳定到 `success≈0.4`
- final 大约 `0.3`

结论：

- `formation` 目前可用，但还没完全修通

## 3. 当前总状态

- `goal_nav`: 已修通
- `risk_nav`: 已修通
- `coverage`: 进入成功区，但未稳定锁住
- `formation`: 接近修通，但 final 稳定性还差一步

## 4. 我现在仍在做什么

当前主线仍然是：

- 继续沿 `coverage success-heavy reinforcement` 推进
- 不再回到那些已经验证收益不高的轻量偏置分支

已经验证收益有限的覆盖分支包括：

- `reward-focus`
- `sector target bias`
- `stronger repulsion`
- `slot-dominant`
- `global slot`

这也是为什么我现在更强调 success-heavy 数据链路，而不是继续盲目堆结构小改。
