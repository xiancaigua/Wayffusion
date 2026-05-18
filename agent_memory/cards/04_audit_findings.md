# 审计发现与问题清单

## 一、可信项

### 1. 核心 benchmark 契约可信

- centralized single-agent 设定清楚
- 统一 task field / agent state / task id / global info 结构清楚
- joint waypoint action 结构清楚
- 4 个任务族与主文档一致

### 2. 训练基础设施可信

- `PPO/SAC/TD3/BC` 训练入口存在并可运行
- 输出目录结构已经稳定到 `timestamp/run_name/checkpoints|snapshot|media`
- 录像、可视化、checkpoint、snapshot 现在构成一套闭环

### 3. 测试面可信

- 当前 `tests/` 下共有 10 个测试文件
- 最近一轮结果是 `14 passed`
- 新增功能已经配了针对性测试，而不是只改代码不加验证

## 二、可复用但要验证的项

### 1. 学习结果摘要

`outputs/eval/` 下存在大量 CSV 和 markdown 摘要，包括：

- `hardening_summary.md`
- `current_stage_summary.md`
- `benchmark_summary.md`
- 多个 scaling / ablation / algorithm comparison CSV

这些结果 **可以作为当前阶段的参考**，但引用前应明确：

- 是 smoke budget 还是长 budget
- 是否已经过最新 calibration
- 是否和当前 checkpoint/snapshot 路径规范一致

### 2. 计划与清单文件

- `PLAN.md`
- `CHECKLIST.md`

它们仍然能反映“large-N calibration round 2”的上下文，但只覆盖了某一阶段，不应把它们当成全项目当前总状态。

## 三、陈旧或冲突项

### 1. `outputs/verification.md` 已陈旧

问题：

- 里面写的是 `6 passed`
- 还引用了旧的 `legacy_mlp_ppo` 路径

而当前实际测试已经是 `14 passed`，目录布局也已升级到 `checkpoints/` 与 `snapshot/`。

结论：

- 该文件只能视为历史参考，不能再作为当前验证基线。

### 2. 配置命名存在重复体系

当前同时存在两套 PPO 命名风格：

- `mlp_ppo.yaml` / `cnn_deepsets_ppo.yaml`
- `ppo_mlp.yaml` / `ppo_cnn_deepsets.yaml`

这会增加以下风险：

- 用户复制命令时选错配置
- agent 在自动发现或比较配置时出现歧义
- 文档统一性下降

### 3. `sitecustomize.py` 有重复副本

当前同时存在：

- 仓库根目录 `sitecustomize.py`
- `scripts/sitecustomize.py`

两者内容相同，都是设置 `KMP_DUPLICATE_LIB_OK=TRUE`。

这不是立即错误，但属于维护噪声，应决定是否保留双副本。

### 4. worktree 当前不是完全干净

审计时可见：

- `.gitignore` 处于修改状态

这不一定是问题，但意味着后续 agent 在做“本轮新增改动”判断时，不能默认当前工作区是全新干净状态。

## 四、潜在工程风险

### 1. 评估入口仍偏手工

- `evaluate_policy.py` 仍要求显式 checkpoint 路径
- `evaluate_scaling.py` 仍要求显式 checkpoint 路径

当前只有 `evaluate_algorithms.py` 做了较好的自动发现兼容。

### 2. 学习有效性不应被过度解读

从现有摘要看：

- benchmark 和 baseline 工程链是可用的
- 但很多学习结果仍然是短预算 / smoke / calibration-stage 级别
- 不能把“脚本可跑”直接当成“方法已学起来”

### 3. 历史输出目录可能混有旧结构

虽然当前代码已经稳定到新目录规范，但 `outputs/` 里仍可能保留旧时期产物：

- 无时间层
- 无 `checkpoints/`
- 无 `snapshot/`

后续 agent 在做统计或自动扫描时必须区分“历史遗留 run”和“当前规范 run”。

## 五、审计结论

项目当前属于：

- `baseline_ready`
- `analysis_ready`
- `usable_training_stack_ready`

但还不是：

- `paper_final_ready`
- `long_budget_result_trusted`

换句话说，工程主干已经可复用，但实验结论层仍需按主题逐项补强。
