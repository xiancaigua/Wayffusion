# Specialist 实验索引

## 已修通任务

### goal_nav

- 最佳目录：
  [outputs/training/bc_ppo/20260528_goalnav_dagger_finetine_ultra_strict/goalnav_dagger_finetine_ultra_strict](/workspace/Wayffusion/Wayffusion/outputs/training/bc_ppo/20260528_goalnav_dagger_finetine_ultra_strict/goalnav_dagger_finetine_ultra_strict)
- 结果：
  - `success_rate_mean=0.8`
  - `goal_coverage_ratio_mean=0.9425`
  - `collision_rate_mean=0.0275`
- 含义：
  - `goal_nav` 已修通

### risk_nav

- 最佳目录：
  [outputs/training/bc_ppo/20260528_phase13_risk_nav_ultra/phase13_risk_nav_ultra](/workspace/Wayffusion/Wayffusion/outputs/training/bc_ppo/20260528_phase13_risk_nav_ultra/phase13_risk_nav_ultra)
- 结果：
  - `success_rate_mean=0.6`
  - `goal_coverage_ratio_mean=0.8292`
  - `collision_rate_mean=0.0488`
- 含义：
  - `risk_nav` 已修通

## 部分修通任务

### coverage

- 当前最佳 PPO 目录：
  [outputs/training/bc_ppo/20260529_phase24_coverage_successonly_v2_bestfinal/phase24_coverage_successonly_v2_bestfinal](/workspace/Wayffusion/Wayffusion/outputs/training/bc_ppo/20260529_phase24_coverage_successonly_v2_bestfinal/phase24_coverage_successonly_v2_bestfinal)
- 当前最佳 BC 目录：
  [outputs/training/bc/20260529_021428/debug_bc_coverage_successonly_coverage_N4_multi_channel_field_plus_task_id](/workspace/Wayffusion/Wayffusion/outputs/training/bc/20260529_021428/debug_bc_coverage_successonly_coverage_N4_multi_channel_field_plus_task_id)
- 当前最佳 best-eval PPO 点：
  - `update=120`
  - `success_rate=0.15`
  - `coverage_ratio≈0.673`
  - `collision≈0.00019`
- 当前最佳 BC 点：
  - `success_rate_mean=0.15`
  - `coverage_ratio_mean=0.6845`
  - `collision_rate_mean=0.0003125`
- 含义：
  - 已进入成功区间，但还没稳定锁住成功

### formation

- 当前最佳 PPO 目录：
  [outputs/training/bc_ppo/20260528_phase14_formation_ultra/phase14_formation_ultra](/workspace/Wayffusion/Wayffusion/outputs/training/bc_ppo/20260528_phase14_formation_ultra/phase14_formation_ultra)
- 当前最佳 BC 目录：
  [outputs/training/bc/20260529_025330/debug_bc_formation_dagger_formation_N4_multi_channel_field_plus_task_id](/workspace/Wayffusion/Wayffusion/outputs/training/bc/20260529_025330/debug_bc_formation_dagger_formation_N4_multi_channel_field_plus_task_id)
- 当前最佳 mid-run PPO 点：
  - `update 40/60/80` 附近 `success≈0.4`
- 当前最好 final：
  - `success_rate_mean≈0.3`
- 含义：
  - 可用，但还没像 `goal_nav/risk_nav` 那样完全修通

## 关键调试阶段

### 早期 sanity / baseline

- `goal_nav baseline PPO`
  [outputs/training/ppo/20260527_132850_goal_nav_baseline/debug_goal_nav_baseline](/workspace/Wayffusion/Wayffusion/outputs/training/ppo/20260527_132850_goal_nav_baseline/debug_goal_nav_baseline)
- `goal_nav controlled PPO`
  [outputs/training/ppo/20260527_132850_goal_nav_controlled/debug_goal_nav_controlled](/workspace/Wayffusion/Wayffusion/outputs/training/ppo/20260527_132850_goal_nav_controlled/debug_goal_nav_controlled)

### goal_nav 修复链

- debug-long 总目录：
  [outputs/debug_long/20260528_035433](/workspace/Wayffusion/Wayffusion/outputs/debug_long/20260528_035433)
- 关键数据：
  - `goal_nav_success_N4_stride2.npz`
  - `goal_nav_dagger_from_bcppo060_plus_success.npz`

### coverage 修复链

- expert-v2 数据：
  [outputs/debug_long/20260528_coverage_expert_v2](/workspace/Wayffusion/Wayffusion/outputs/debug_long/20260528_coverage_expert_v2)
- DAgger 数据：
  [outputs/debug_long/20260528_coverage_dagger](/workspace/Wayffusion/Wayffusion/outputs/debug_long/20260528_coverage_dagger)
- success-policy 数据：
  [outputs/debug_long/20260529_coverage_success_policy](/workspace/Wayffusion/Wayffusion/outputs/debug_long/20260529_coverage_success_policy)

### risk_nav / formation 数据

- risk_nav success dataset：
  [outputs/debug_long/20260528_risk_nav_success](/workspace/Wayffusion/Wayffusion/outputs/debug_long/20260528_risk_nav_success)
- formation success dataset：
  [outputs/debug_long/20260528_formation_success](/workspace/Wayffusion/Wayffusion/outputs/debug_long/20260528_formation_success)
- formation success-policy dataset：
  [outputs/debug_long/20260528_formation_success_policy](/workspace/Wayffusion/Wayffusion/outputs/debug_long/20260528_formation_success_policy)

## 图表位置

关键 run 的图表会保存在各自目录下的：

- `plots/`

如果某个目录下有：

- `eval_success_rate.png`
- `eval_reward.png`
- `eval_collision_rate.png`
- `final_eval_summary.png`

就说明这条 run 已经被整理成可直接看的结果。

当前已经整理过图表的关键目录包括：

- [outputs/training/ppo/20260527_132850_goal_nav_baseline/debug_goal_nav_baseline/plots](/workspace/Wayffusion/Wayffusion/outputs/training/ppo/20260527_132850_goal_nav_baseline/debug_goal_nav_baseline/plots)
- [outputs/training/ppo/20260527_132850_goal_nav_controlled/debug_goal_nav_controlled/plots](/workspace/Wayffusion/Wayffusion/outputs/training/ppo/20260527_132850_goal_nav_controlled/debug_goal_nav_controlled/plots)
- [outputs/training/bc_ppo/20260528_goalnav_dagger_finetune_ultra_strict/goalnav_dagger_finetine_ultra_strict/plots](/workspace/Wayffusion/Wayffusion/outputs/training/bc_ppo/20260528_goalnav_dagger_finetine_ultra_strict/goalnav_dagger_finetine_ultra_strict/plots)
- [outputs/training/bc_ppo/20260528_phase13_risk_nav_ultra/phase13_risk_nav_ultra/plots](/workspace/Wayffusion/Wayffusion/outputs/training/bc_ppo/20260528_phase13_risk_nav_ultra/phase13_risk_nav_ultra/plots)
- [outputs/training/bc_ppo/20260528_phase14_formation_ultra/phase14_formation_ultra/plots](/workspace/Wayffusion/Wayffusion/outputs/training/bc_ppo/20260528_phase14_formation_ultra/phase14_formation_ultra/plots)
- [outputs/training/bc_ppo/20260528_phase21_coverage_bestfinal/phase21_coverage_bestfinal/plots](/workspace/Wayffusion/Wayffusion/outputs/training/bc_ppo/20260528_phase21_coverage_bestfinal/phase21_coverage_bestfinal/plots)
- [outputs/training/bc_ppo/20260529_phase24_coverage_successonly_v2_bestfinal/phase24_coverage_successonly_v2_bestfinal/plots](/workspace/Wayffusion/Wayffusion/outputs/training/bc_ppo/20260529_phase24_coverage_successonly_v2_bestfinal/phase24_coverage_successonly_v2_bestfinal/plots)
- [outputs/training/bc/20260529_021428/debug_bc_coverage_successonly_coverage_N4_multi_channel_field_plus_task_id/plots](/workspace/Wayffusion/Wayffusion/outputs/training/bc/20260529_021428/debug_bc_coverage_successonly_coverage_N4_multi_channel_field_plus_task_id/plots)
- [outputs/training/bc/20260529_025330/debug_bc_formation_dagger_formation_N4_multi_channel_field_plus_task_id/plots](/workspace/Wayffusion/Wayffusion/outputs/training/bc/20260529_025330/debug_bc_formation_dagger_formation_N4_multi_channel_field_plus_task_id/plots)
