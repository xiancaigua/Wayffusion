# 系统契约与核心设计

## 问题设定

本仓库的核心问题设定是：

- 把整个 UAV swarm 视为 **一个 centralized agent**
- 输入是全局任务场、全体 agent 状态和任务标识
- 输出是所有无人机的 joint waypoint / subgoal 动作
- 主目标是 benchmark 与 learning baseline 验证，不是 MARL、真实飞控或高保真仿真

## 主接口

环境主接口是 Gymnasium-style single-agent：

- `reset() -> obs`
- `step(action) -> obs, reward, terminated, truncated, info`

观测结构固定为：

- `task_field`
- `agents`
- `task_id`
- `global_info`

动作结构固定为：

- joint waypoint delta，形状 `[N, 2]`

## 统一任务场

`task_field` 固定 9 通道：

1. obstacle
2. goal_reward
3. target_probability
4. desired_occupancy
5. risk
6. visited
7. agent_density
8. communication_quality
9. formation_template

任务可以只激活部分通道，但 observation 结构不随任务改变。

## 任务族

当前任务族是 4 个：

- `goal_nav`
- `coverage`
- `formation`
- `risk_nav`

这 4 个任务通过共享环境、共享 task-field 和共享 action interface 接入，而不是 4 套独立环境。

## policy 家族

当前策略结构包括：

- `MLPPolicy`
  - 仅适合 fixed-N 小规模
- `CNNDeepSetsPolicy`
  - 当前 variable-N 主路线
- `CNNAttentionPolicy`
  - 当前可运行，但不是默认主线

## 学习算法家族

当前仓库包含：

- `BC`
- `PPO`
- `SAC`
- `TD3`

统一前提是：

- centralized single-policy
- 不做 MARL 主接口
- 不做 per-agent actor
- 不做 MAPPO

## 当前可信结论

- centralized single-agent benchmark 的问题边界是清晰的
- observation / action contract 在代码层已固定
- policy / algorithm / tasks 的基本路由与 README 一致
- 这个项目已经从最初的 smoke benchmark 进入了 learning baseline + scaling 的工程阶段
