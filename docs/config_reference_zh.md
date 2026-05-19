# 配置文件说明（中文）

本文档面向 Wayffusion 当前仓库内的配置体系，重点回答三件事：

1. 不同训练/评估脚本分别应该读哪一类配置；
2. 每一大类配置里有哪些参数、每个参数的含义是什么；
3. 每一类配置从哪个模范文件开始改最稳妥。

模范文件统一放在：

- `configs/examples/`

## 1. 配置类别总览

Wayffusion 当前有三大类配置：

| 大类 | 目录 | 作用 |
|---|---|---|
| 环境配置 | `configs/env/` | 定义地图、任务、reward、观察形式、scaling 行为 |
| 策略/算法配置 | `configs/policy/` | 定义网络结构和训练超参数 |
| 评估协议配置 | `configs/eval/` | 定义 baseline、scaling、ablation、算法对比的评估协议 |

## 2. 不同脚本应该看哪类配置

### 2.1 训练脚本

| 脚本 | 主要读取的配置 |
|---|---|
| `scripts/train_bc.py` | `configs/policy/bc_*.yaml` |
| `scripts/train_ppo.py` | `configs/policy/ppo_*.yaml`、`configs/policy/ppo_from_bc*.yaml` |
| `scripts/train_sac.py` | `configs/policy/sac_*.yaml` |
| `scripts/train_td3.py` | `configs/policy/td3_*.yaml` |
| `scripts/run_multitask_ppo_20k.ps1` | 默认调用 `configs/policy/ppo_cnn_deepsets_multitask_20k.yaml` |

训练脚本通常还会结合：

- `--env-config configs/env/*.yaml`
- `--tasks ...`
- `--agent_counts ...`
- `--scaling_mode ...`
- `--obs_variant ...`

也就是说，训练脚本的核心是：

- 一个 `policy config`
- 一个 `env config`
- 若干命令行覆盖项

### 2.2 评估脚本

| 脚本 | 主要读取的配置 |
|---|---|
| `scripts/evaluate_baselines.py` | `configs/eval/eval_single_task.yaml`、`eval_multitask.yaml`、`eval_generalization.yaml` |
| `scripts/evaluate_policy.py` | `--policy-config` + `--checkpoint`，可配合 `configs/env/*.yaml` |
| `scripts/evaluate_scaling.py` | `configs/eval/eval_scaling_fixed_N.yaml`、`eval_scaling_variable_N.yaml` |
| `scripts/evaluate_algorithms.py` | `configs/eval/eval_algorithm_comparison.yaml` |

### 2.3 推荐起手顺序

如果你要开始一个新实验，推荐顺序是：

1. 先从 `configs/examples/env_template.yaml` 改环境；
2. 再从对应算法的 `configs/examples/policy_*_template.yaml` 改训练超参数；
3. 最后从一个 `configs/examples/eval_*_template.yaml` 改评估协议。

## 3. 环境配置：`configs/env/`

### 3.1 常见文件

| 文件 | 用途 |
|---|---|
| `base.yaml` | 所有环境参数的完整基础版本 |
| `goal_nav.yaml` | 单任务 `goal_nav` 覆盖配置 |
| `coverage.yaml` | 单任务 `coverage` 覆盖配置 |
| `formation.yaml` | 单任务 `formation` 覆盖配置 |
| `risk_nav.yaml` | 单任务 `risk_nav` 覆盖配置 |
| `multitask.yaml` | 多任务训练/评估常用环境配置 |
| `agents_4.yaml` ~ `agents_100.yaml` | 固定 agent 数量覆盖配置 |
| `scaling.yaml` | agent scaling 辅助配置 |

### 3.2 顶层通用参数

| 参数 | 含义 |
|---|---|
| `seed` | 默认随机种子 |
| `task_name` | 固定单任务名；为 `null` 时由 `task_names` 采样 |
| `task_names` | 可被采样的任务集合 |
| `task_sampling_probs` | 多任务采样概率 |
| `map_size` | 基础地图边长；`density_preserving` 下会再缩放 |
| `reference_num_agents` | scaling 参考 agent 数，一般为 4 |
| `grid_size` | task field / 栅格可视化的分辨率 |
| `num_agents` | 环境中的 UAV 数量 |
| `scaling_mode` | `fixed_map` 或 `density_preserving` |
| `scale_max_steps_with_map` | 地图放大后是否同步放大 `max_steps` |
| `max_steps` | 单个 episode 的最大步数 |
| `dt` | 低层动力学时间步长 |
| `kp` | waypoint 控制器比例增益 |
| `max_speed` | UAV 最大速度 |
| `max_waypoint_step` | 每步最大相对 waypoint 增量 |
| `collision_radius` | 机间碰撞判定半径 |
| `goal_radius` | 到达目标的判定半径 |
| `coverage_radius` | coverage 任务的覆盖半径 |
| `formation_radius` | formation 目标半径 |
| `formation_tolerance` | formation 成功判定容差 |
| `obstacle_density` | 障碍物密度 |
| `obstacle_size_range` | 障碍物矩形尺寸范围 |
| `risk_blob_count` | 风险场 blob 数量 |
| `risk_blob_sigma` | 风险场 blob 宽度 |
| `no_fly_threshold` | 超过该阈值视为禁飞区/硬约束违规 |
| `target_motion` | formation 目标运动模式，如 `static` 或 `linear` |
| `communication_decay` | 通信质量场衰减尺度 |
| `render_dpi` | 渲染输出 DPI |

### 3.3 观察相关参数

| 参数 | 含义 |
|---|---|
| `observation_mode` | `multi_channel` / `multi_channel_field` / `single_channel_field` / `no_spatial_field` |
| `include_task_id` | 是否保留 one-hot `task_id` |
| `include_agent_density` | 是否保留 `agent_density` 通道 |
| `drop_channels` | 强制置零的 field channel 列表 |
| `single_channel_weights` | 压缩为单通道时的加权系数 |
| `field_channels` | 固定通道顺序说明，用于对齐和可视化 |

兼容性说明：

- `task_id_only` 仍可作为旧 alias 使用；
- 但文档和新实验命名统一推荐 `no_spatial_field`；
- `no_spatial_field` 的真实语义是：`task_field` 全零，但 `agents`、`task_id`、`global_info` 仍然存在。

### 3.4 任务子配置

#### `goal_nav`

| 参数 | 含义 |
|---|---|
| `num_goals_range` | 目标点数量范围 |
| `density_goal_scale_exponent` | `density_preserving` 下 goal 数增长指数 |

#### `coverage`

| 参数 | 含义 |
|---|---|
| `multi_peak_probability` | 是否生成多峰目标概率场 |
| `success_ratio` | coverage 成功阈值 |
| `density_peak_scale_exponent` | `density_preserving` 下 peak 数增长指数 |

#### `formation`

| 参数 | 含义 |
|---|---|
| `train_templates` | 训练阶段允许的 formation 模板 |
| `eval_templates` | 测试/泛化阶段允许的 formation 模板 |
| `angular_tolerance` | 角度均匀性容差 |

#### `risk_nav`

| 参数 | 含义 |
|---|---|
| `num_goals_range` | risk-aware 导航目标数范围 |
| `risk_weight_scale` | 风险场权重调节因子 |
| `density_goal_scale_exponent` | `density_preserving` 下 goal 数增长指数 |

### 3.5 `task_sets`

`task_sets` 用于给常用任务组合起标准别名，便于脚本和实验表格复用，例如：

- `single_goal_nav`
- `two_goal_cov`
- `three_goal_cov_form`
- `four_task`

### 3.6 `reward_weights`

#### `common`

| 参数 | 含义 |
|---|---|
| `collision` | pairwise collision 惩罚 |
| `obstacle_collision` | 障碍碰撞惩罚 |
| `path_length` | 路径长度惩罚 |
| `time` | 每步时间惩罚 |
| `safety_violation` | 禁飞区违规惩罚 |
| `risk` | 风险暴露惩罚 |

#### `goal_nav`

| 参数 | 含义 |
|---|---|
| `progress` | 目标推进奖励 |
| `goal_reached` | 新完成目标奖励 |
| `repeated_goal` | 重复占用已完成目标惩罚 |

#### `coverage`

| 参数 | 含义 |
|---|---|
| `new_coverage` | 新覆盖区域奖励 |
| `high_probability` | 覆盖高概率区域奖励 |
| `repeated_coverage` | 重复覆盖惩罚 |

#### `formation`

| 参数 | 含义 |
|---|---|
| `error_reduction` | formation error 改善奖励 |
| `angular_coverage` | 角度覆盖均匀性奖励 |
| `radius_error_penalty` | 半径误差惩罚系数，正数 |
| `stability` | 短窗口稳定性奖励 |

兼容性说明：

- 旧字段 `radius_error` 仍会被代码兼容读取；
- 新配置统一推荐写成 `radius_error_penalty`。

#### `risk_nav`

| 参数 | 含义 |
|---|---|
| `progress` | 主任务推进奖励 |
| `goal_reached` | 到达目标奖励 |
| `risk_exposure` | 风险暴露惩罚 |

### 3.7 环境模范文件

- [`configs/examples/env_template.yaml`](../configs/examples/env_template.yaml)

## 4. 策略 / 算法配置：`configs/policy/`

### 4.1 共用参数

| 参数 | 含义 |
|---|---|
| `name` | 当前 run 的逻辑名称，会进入输出目录名 |
| `algorithm` / `type` | 算法类别标记；仓库里两种写法都存在 |
| `policy_class` | `mlp` / `cnn_deepsets` / `attention` |

### 4.2 架构参数

#### `mlp`

| 参数 | 含义 |
|---|---|
| `hidden_dims` | MLP 隐层宽度列表 |

#### `cnn_deepsets`

| 参数 | 含义 |
|---|---|
| `cnn_channels` | field CNN 每层输出通道数 |
| `agent_hidden_dim` | 单个 agent token 的共享编码维度 |
| `joint_hidden_dim` | 群体上下文融合维度 |

#### `attention`

| 参数 | 含义 |
|---|---|
| `cnn_channels` | field CNN 编码器通道数 |
| `embed_dim` | token 嵌入维度 |
| `num_heads` | 多头注意力头数 |
| `num_layers` | 注意力编码层数 |

### 4.3 Baseline policy 配置

适用文件：

- `random.yaml`
- `heuristic.yaml`

参数：

| 参数 | 含义 |
|---|---|
| `name` | baseline 名称 |
| `type` | baseline 类别标记 |
| `policy_name` | `random` 或 `heuristic` |

模范文件：

- [`configs/examples/policy_baseline_template.yaml`](../configs/examples/policy_baseline_template.yaml)

### 4.4 BC 配置

适用文件：

- `bc_mlp.yaml`
- `bc_cnn_deepsets.yaml`
- `bc_attention.yaml`

参数：

| 参数 | 含义 |
|---|---|
| `batch_size` | DataLoader batch size |
| `epochs` | 监督训练轮数 |
| `learning_rate` | 学习率 |
| `eval_interval` | 当前主要用于配置对齐和后续扩展的保留字段 |

说明：

- `train_bc.py` 主要读取这一类配置；
- 如果想做 variable-`N` 训练，推荐优先用 `cnn_deepsets` 或 `attention`，不要用 `mlp`。

模范文件：

- [`configs/examples/policy_bc_template.yaml`](../configs/examples/policy_bc_template.yaml)

### 4.5 PPO / BC+PPO 配置

适用文件：

- `ppo_mlp.yaml`
- `ppo_cnn_deepsets.yaml`
- `ppo_attention.yaml`
- `ppo_from_bc.yaml`
- `ppo_cnn_deepsets_multitask_20k.yaml`

参数：

| 参数 | 含义 |
|---|---|
| `num_envs` | 同步 vectorized env 数量 |
| `rollout_steps` | 每次 rollout 的采样步数 |
| `total_updates` | PPO 更新轮数 |
| `target_episodes` | episode 停止条件；可选 |
| `epochs` | 每个 PPO batch 的优化轮数 |
| `minibatch_size` | PPO update 的 mini-batch 大小 |
| `gamma` | 折扣因子 |
| `gae_lambda` | GAE 参数 |
| `clip_coef` | PPO clipping 系数 |
| `ent_coef` | 熵正则系数 |
| `vf_coef` | value loss 系数 |
| `learning_rate` | 学习率 |
| `max_grad_norm` | 梯度裁剪阈值 |
| `reward_norm` | 是否开启 reward normalization |
| `advantage_norm` | 是否标准化 advantage |
| `lr_schedule` | 学习率调度，常见为 `constant` 或 `linear` |
| `eval_interval` | 每多少个 update 做一次 eval / checkpoint |

说明：

- `ppo_from_bc*.yaml` 本质上仍然是 PPO 配置；
- 区别只是在调用 `train_ppo.py` 时额外传 `--init_checkpoint`；
- 新实验优先推荐 `ppo_*` 命名，旧的 `mlp_ppo.yaml` / `cnn_deepsets_ppo.yaml` 视为兼容别名。

模范文件：

- [`configs/examples/policy_ppo_template.yaml`](../configs/examples/policy_ppo_template.yaml)

### 4.6 SAC / BC+SAC 配置

适用文件：

- `sac_cnn_deepsets.yaml`
- `sac_from_bc.yaml`

参数：

| 参数 | 含义 |
|---|---|
| `batch_size` | replay sample batch size |
| `num_envs` | 并行环境数 |
| `replay_size` | replay buffer 容量 |
| `warmup_steps` | 随机动作 warmup 步数 |
| `total_steps` | 总环境交互步数 |
| `eval_interval_steps` | 每隔多少环境步做一次 eval |
| `gamma` | 折扣因子 |
| `tau` | target network 软更新系数 |
| `learning_rate` | actor / critic 学习率 |
| `policy_delay` | 当前配置里保留的兼容字段，当前 SAC trainer 不显式使用 |
| `target_entropy_scale` | 当前配置里保留的扩展字段，当前 trainer 不显式使用 |

模范文件：

- [`configs/examples/policy_sac_template.yaml`](../configs/examples/policy_sac_template.yaml)

### 4.7 TD3 配置

适用文件：

- `td3_cnn_deepsets.yaml`

参数：

| 参数 | 含义 |
|---|---|
| `batch_size` | replay sample batch size |
| `num_envs` | 并行环境数 |
| `replay_size` | replay buffer 容量 |
| `warmup_steps` | 随机动作 warmup 步数 |
| `total_steps` | 总环境交互步数 |
| `eval_interval_steps` | 每隔多少环境步做一次 eval |
| `gamma` | 折扣因子 |
| `tau` | target network 软更新系数 |
| `learning_rate` | 学习率 |
| `policy_delay` | actor 延迟更新间隔 |
| `policy_noise` | target policy smoothing 噪声标准差 |
| `noise_clip` | smoothing 噪声裁剪范围 |

模范文件：

- [`configs/examples/policy_td3_template.yaml`](../configs/examples/policy_td3_template.yaml)

## 5. 评估配置：`configs/eval/`

### 5.1 Baseline 协议类

#### `eval_single_task.yaml`

| 参数 | 含义 |
|---|---|
| `mode` | 固定为 `single_task` |
| `env_config` | 环境配置路径 |
| `tasks` | 要分别评估的任务列表 |
| `policies` | baseline 列表，通常为 `heuristic` / `random` |
| `episodes_per_task` | 每个任务每个 baseline 跑多少回合 |
| `save_rollouts` | 是否保存代表性 rollout 图 |
| `output_dir` | 输出目录 |

模范文件：

- [`configs/examples/eval_single_task_template.yaml`](../configs/examples/eval_single_task_template.yaml)

#### `eval_multitask.yaml`

| 参数 | 含义 |
|---|---|
| `mode` | 固定为 `multitask` |
| `env_config` | 环境配置路径 |
| `policies` | baseline 列表 |
| `episodes` | 总 rollout 数 |
| `save_rollouts` | 是否保存代表性 rollout |
| `output_dir` | 输出目录 |

模范文件：

- [`configs/examples/eval_multitask_template.yaml`](../configs/examples/eval_multitask_template.yaml)

#### `eval_generalization.yaml`

| 参数 | 含义 |
|---|---|
| `mode` | 固定为 `generalization` |
| `train_env_config` | 训练分布环境覆盖项 |
| `test_env_config` | 测试分布环境覆盖项 |
| `policies` | baseline 列表 |
| `episodes` | 每个 split 的 rollout 数 |
| `output_dir` | 输出目录 |

模范文件：

- [`configs/examples/eval_generalization_template.yaml`](../configs/examples/eval_generalization_template.yaml)

### 5.2 学习基线 / scaling / ablation 类

#### `eval_learning_baselines.yaml`

| 参数 | 含义 |
|---|---|
| `tasks` | 任务集合 |
| `agent_counts` | 要评估的 agent 数量 |
| `algorithms` | 算法列表 |
| `architectures` | 要比较的网络结构 |
| `episodes` | 每个设置评估回合数 |
| `scaling_mode` | 地图缩放模式 |
| `output_path` | 输出 CSV 路径 |

模范文件：

- [`configs/examples/eval_learning_baselines_template.yaml`](../configs/examples/eval_learning_baselines_template.yaml)

#### `eval_scaling_fixed_N.yaml`

| 参数 | 含义 |
|---|---|
| `protocol` | 固定为 `fixed_N` |
| `tasks` | 任务集合 |
| `agent_counts` | 要逐个评估的 `N` |
| `episodes` | 每个 `N` 的评估回合数 |
| `scaling_mode` | `fixed_map` / `density_preserving` |
| `output_path` | 输出 CSV 路径 |

模范文件：

- [`configs/examples/eval_scaling_fixed_n_template.yaml`](../configs/examples/eval_scaling_fixed_n_template.yaml)

#### `eval_scaling_variable_N.yaml`

| 参数 | 含义 |
|---|---|
| `protocol` | 固定为 `variable_N` |
| `train_sets` | 训练阶段的 agent-count 集合定义 |
| `test_agent_counts` | 测试阶段的 `N` 列表 |
| `episodes` | 每个测试 `N` 的评估回合数 |
| `scaling_mode` | 地图缩放模式 |
| `output_path` | 输出 CSV 路径 |

模范文件：

- [`configs/examples/eval_scaling_variable_n_template.yaml`](../configs/examples/eval_scaling_variable_n_template.yaml)

#### `eval_ablation_observation.yaml`

| 参数 | 含义 |
|---|---|
| `tasks` | 任务集合 |
| `agent_counts` | 要评估的 agent 数量 |
| `observation_modes` | 观察消融配置列表 |
| `episodes` | 每个设置评估回合数 |
| `output_path` | 输出 CSV 路径 |

模范文件：

- [`configs/examples/eval_ablation_observation_template.yaml`](../configs/examples/eval_ablation_observation_template.yaml)

#### `eval_ablation_architecture.yaml`

| 参数 | 含义 |
|---|---|
| `tasks` | 任务集合 |
| `agent_counts` | 要评估的 agent 数量 |
| `architectures` | 架构列表 |
| `episodes` | 每个设置评估回合数 |
| `output_path` | 输出 CSV 路径 |

模范文件：

- [`configs/examples/eval_ablation_architecture_template.yaml`](../configs/examples/eval_ablation_architecture_template.yaml)

#### `eval_algorithm_comparison.yaml`

| 参数 | 含义 |
|---|---|
| `tasks` | 任务集合 |
| `agent_counts` | 要评估的 agent 数量 |
| `algorithms` | 算法列表 |
| `episodes` | 每个设置评估回合数 |
| `scaling_mode` | 地图缩放模式 |
| `output_path` | 输出 CSV 路径 |

模范文件：

- [`configs/examples/eval_algorithm_comparison_template.yaml`](../configs/examples/eval_algorithm_comparison_template.yaml)

## 6. 模范文件总表

| 类别 | 模范文件 |
|---|---|
| 环境 | [`configs/examples/env_template.yaml`](../configs/examples/env_template.yaml) |
| Baseline policy | [`configs/examples/policy_baseline_template.yaml`](../configs/examples/policy_baseline_template.yaml) |
| BC | [`configs/examples/policy_bc_template.yaml`](../configs/examples/policy_bc_template.yaml) |
| PPO / BC+PPO | [`configs/examples/policy_ppo_template.yaml`](../configs/examples/policy_ppo_template.yaml) |
| SAC / BC+SAC | [`configs/examples/policy_sac_template.yaml`](../configs/examples/policy_sac_template.yaml) |
| TD3 | [`configs/examples/policy_td3_template.yaml`](../configs/examples/policy_td3_template.yaml) |
| 单任务 baseline 评估 | [`configs/examples/eval_single_task_template.yaml`](../configs/examples/eval_single_task_template.yaml) |
| 多任务 baseline 评估 | [`configs/examples/eval_multitask_template.yaml`](../configs/examples/eval_multitask_template.yaml) |
| 泛化 baseline 评估 | [`configs/examples/eval_generalization_template.yaml`](../configs/examples/eval_generalization_template.yaml) |
| 学习基线对比 | [`configs/examples/eval_learning_baselines_template.yaml`](../configs/examples/eval_learning_baselines_template.yaml) |
| fixed-N scaling | [`configs/examples/eval_scaling_fixed_n_template.yaml`](../configs/examples/eval_scaling_fixed_n_template.yaml) |
| variable-N scaling | [`configs/examples/eval_scaling_variable_n_template.yaml`](../configs/examples/eval_scaling_variable_n_template.yaml) |
| 观察消融 | [`configs/examples/eval_ablation_observation_template.yaml`](../configs/examples/eval_ablation_observation_template.yaml) |
| 架构消融 | [`configs/examples/eval_ablation_architecture_template.yaml`](../configs/examples/eval_ablation_architecture_template.yaml) |
| 算法对比 | [`configs/examples/eval_algorithm_comparison_template.yaml`](../configs/examples/eval_algorithm_comparison_template.yaml) |

## 7. 兼容性提醒

- `task_id_only` 仍可用，但新实验请统一改成 `no_spatial_field`
- `reward_weights.formation.radius_error` 是旧写法，新配置统一写成 `radius_error_penalty`
- `configs/policy/` 中仍存在 `mlp_ppo.yaml` 和 `cnn_deepsets_ppo.yaml` 这类旧命名；新实验优先使用 `ppo_*` 这一套命名

## 8. 一句话建议

如果你只想稳妥开始一个新实验：

1. 复制一个 `env_template.yaml`
2. 再复制一个与你算法对应的 `policy_*_template.yaml`
3. 最后复制一个与你评估协议对应的 `eval_*_template.yaml`

这样最不容易因为漏参数或命名不一致而踩坑。
