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

1. Keep `goal_nav` factorized-group as the first validated new-architecture specialist:
   `outputs/training/bc_ppo/20260530_phase36_goalnav_factorized_group_ppo/phase36_goalnav_factorized_group_ppo/checkpoints/checkpoint_best_eval.pt`.
2. Coverage decision point now has two branches:
   old spatial-head h300 best: `success_rate_mean=0.56`;
   new factorized-group h300 best: `success_rate_mean=0.40`.
3. If the project priority is “new architecture everywhere”, tune factorized-group coverage next:
   lower collision and path inefficiency, likely via fewer groups, stronger group waypoint bias, or group-specific suppression.
4. If the priority is “best specialist now”, keep old phase34 as coverage canonical and continue factorized-group only as a replacement branch.
5. h200 coverage is still unresolved regardless of architecture; both old and new policies remain far below a strong specialist under the 200-step budget.
6. Do not use `coverage_expert_v4` or the band-sweep diagnostic as teachers; both had `success_rate=0.0`.
7. Run formation best-checkpoint stabilization after the coverage architecture decision.
8. Refresh `outputs/verification.md`, checkpoint latest helpers, and canonical multitask runs after specialist routes stabilize.
