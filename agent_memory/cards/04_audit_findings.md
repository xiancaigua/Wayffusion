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
- 最近针对 goal_nav/field/reward/action-distribution 的一轮结果是 `24 passed`
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

### 0. 服务器训练依赖与部署文档缺口已处理

2026-05-20 服务器训练适配已处理以下工程缺口：

- `utils/profiling.py` 依赖 `psutil`，现在 `requirements.txt` 和 `requirements-server.txt` 都显式包含 `psutil`
- 新增 `requirements-server.txt`，用于已有 CUDA torch 的 PyTorch Docker 镜像，避免服务器上 pip 安装时升级 torch
- 新增 `docs/server_training_zh.md` 和 `scripts/server/`，覆盖 headless 训练、代理验证、GPU 检查、PPO smoke train、TensorBoard 启动和 SSH 转发

剩余注意事项：

- `requirements.txt` 仍保留通用环境的 `torch>=2.9` 约束；服务器容器应使用 `requirements-server.txt`
- 服务器训练文档是部署流程来源，不改变算法、环境、任务、策略或输出目录语义

### 1. 评估入口仍偏手工

- `evaluate_policy.py` 仍要求显式 checkpoint 路径
- `evaluate_scaling.py` 仍要求显式 checkpoint 路径

当前只有 `evaluate_algorithms.py` 做了较好的自动发现兼容。

### 2. 学习有效性不应被过度解读

从现有摘要看：

- benchmark 和 baseline 工程链是可用的
- 但很多学习结果仍然是短预算 / smoke / calibration-stage 级别
- 不能把“脚本可跑”直接当成“方法已学起来”

### 3. 当前 goal_nav debug 结论

从最近一轮长 debug 看：

- 随机初始化 PPO 在 goal_nav 上长期维持 `success_rate=0`
- 仅修正 `goal_reward` 为剩余目标后，PPO 仍然没有学出稳定的 deterministic 目标追踪
- `goal_progress` 和 `distance_penalty` 改为 remaining-goal shaping 后，PPO 出现了更好的中期信号，但 success 仍然不稳定
- 成功轨迹 BC 能把 deterministic action 从接近 0 拉到非零、并提升 goal alignment，但 collision 偏高
- BC + conservative PPO 是第一个出现非零、可复现 success 的组合，但还没有达到稳定收敛标准
- DAgger-style learner-state relabeling 已经被加入下一步修复路径，因为当前主要问题更像是分布偏移和安全性，而不是单纯 reward 尺度

更新：

- 基于 `goal_nav_dagger_from_bcppo060_plus_success.npz` 的 DAgger BC 已经把 specialist 拉到高成功区：
  - `success_rate_mean=0.8`
  - `goal_coverage_ratio_mean=0.9125`
  - `collision_rate_mean=0.0646`
- 在该 DAgger BC checkpoint 上继续做 ultra-strict PPO fine-tune 后，goal_nav 当前已经进入可认为“稳定收敛”的区间：
  - `outputs/training/bc_ppo/20260528_goalnav_dagger_finetune_ultra_strict/...`
  - final `success_rate_mean=0.8`
  - final `goal_coverage_ratio_mean=0.9425`
  - final `collision_rate_mean=0.0275`
  - final `return_mean=10.1225`

结论：

- `goal_nav` 当前主线已经不是“未修通”，而是“已修通且可复用”
- 后续重点应该从 `goal_nav` 转移到 `coverage`

### 3. 历史输出目录可能混有旧结构

虽然当前代码已经稳定到新目录规范，但 `outputs/` 里仍可能保留旧时期产物：

- 无时间层
- 无 `checkpoints/`
- 无 `snapshot/`

### 4. debug-long 结果不应被误读为最终收敛

最近的 `outputs/debug_long/20260528_035433/` 记录显示：

- goal_nav 的初始 PPO 仍然失败
- success-only BC 和 BC+PPO 都有改善，但 collision 仍高
- 当前还没有满足“连续若干次 evaluation 中 success 不接近 0 且 collision 不恶化”的最终标准

结论：

- 这些结果是重要的诊断信号，不是最终收敛证据

### 5. 当前 coverage debug 结论

- 低熵 controlled PPO 能稳定学到非随机的覆盖行为，当前最强切片大约达到：
  - `coverage_ratio≈0.45`
  - `eval_reward≈-0.55`
  - `collision_rate≈0.08~0.12`
- 但当前还没有任何 coverage specialist 过 `success_ratio=0.82` 门槛
- heuristic BC coverage probe 没有优于 controlled PPO 主线：
  - final `coverage_ratio_mean≈0.406`
  - final `success_rate_mean=0.0`
  - final `collision_rate_mean≈0.211`
- 当前最值得继续的 coverage 分支已经切换为：
  - `outputs/training/bc_ppo/20260528_phase4_coverage_creditassign/phase4_coverage_creditassign/`
  - update 20 时 `eval_reward≈-0.551`, `coverage_ratio≈0.450`, `collision≈0.0899`
  - update 40 时 `eval_reward≈-1.008`, `coverage_ratio≈0.449`, `collision≈0.0944`
- 新增的 `coverage expert-v2` 分支已经进一步提高了 warm-start 上限：
  - expert-v2 quick probe: `coverage_ratio≈0.579`, `collision≈0.0865`
  - BC on expert-v2 dataset: `coverage_ratio_mean≈0.512`, `collision_rate_mean≈0.150`, `success_rate_mean=0.0`
  - plain strict PPO fine-tune from that BC checkpoint still degraded to `coverage_ratio≈0.486`
- 新增的 repulsion PPO 分支首次把 coverage specialist 明显推过旧平台：
  - `outputs/training/bc_ppo/20260528_phase8_coverage_repulsion/phase8_coverage_repulsion/`
  - update 20: `coverage_ratio≈0.578`
  - update 20: `collision≈0.0015`
  - update 20: `success=0.0`
  - update 40: `coverage_ratio≈0.664`, `success≈0.05`
  - update 120 final eval still only `success=0.0`, but `coverage_ratio≈0.628` and `collision≈0.00075`
- repulsion + reward-focus combined run did not materially beat the plain repulsion branch:
  - `outputs/training/bc_ppo/20260528_phase9_coverage_repulsion_rewardfocus/phase9_coverage_repulsion_rewardfocus/`
  - update 20: `coverage_ratio≈0.665`, `success=0.0`
- spatial-action-head coverage branch now exceeds the plain repulsion branch in stability:
  - BC warm-start:
    - `coverage_ratio_mean≈0.661`
    - `success_rate_mean≈0.05`
    - `collision_rate_mean≈0.00094`
  - PPO fine-tune:
    - update 20: `success≈0.10`, `coverage_ratio≈0.667`, `collision≈0.000625`
    - update 40: `success≈0.10`, `coverage_ratio≈0.670`, `collision≈0.0003125`
    - update 60: `success≈0.10`, `coverage_ratio≈0.682`, `collision≈0.00025`
    - final eval: `success=0.0`, `coverage_ratio≈0.667`, `collision≈0.003`, `return≈8.083`
- spatial-target suppression probe did not improve on that branch:
  - update 20: `success≈0.10`, `coverage_ratio≈0.655`
  - update 40: `success≈0.05`, with reward and stability both below the no-suppression branch

结论：

- coverage 当前的主问题不是“完全不会动”，而是“已经学到部分覆盖结构，但还没有足够长期、足够全局的完成行为”
- canonical heuristic BC 当前不是主线
- 当前 coverage 最有希望的方向已经更新为：`expert-v2 dataset -> spatial-head BC -> spatial-head PPO`
- coverage 还没有达到 `success_ratio=0.82`，但已经首次在连续多个 eval 点上保持非零 success，并且碰撞率极低
- 换句话说，coverage 已从“完全不收敛”进入了“阶段性可收敛、但终局稳定性仍不足”的状态

### 6. 当前 risk_nav debug 结论

- risk-aware heuristic 已经明显优于随机，且与 goal_nav 的结构相近
- 用 full-state heuristic dataset 做 BC 后，risk_nav specialist 已经提升到可用区间
- ultra-strict PPO 从该 BC checkpoint 出发后，没有出现“fine-tune 一上来就崩”的问题
- 当前 final line：
  - `outputs/training/bc_ppo/20260528_phase13_risk_nav_ultra/phase13_risk_nav_ultra/`
  - final `success_rate_mean=0.6`
  - final `collision_rate_mean≈0.049`
  - final `goal_coverage_ratio_mean≈0.829`

结论：

- `risk_nav` 当前可以视为已修通

### 7. 当前 formation debug 结论

- heuristic baseline 本身较强：`success_rate≈0.4`
- success-only BC 不够，full-state BC 也没有显著超过 heuristic
- ultra-strict PPO 从 full-state BC checkpoint 出发后，第一次在多个 eval 点上稳定打到 heuristic 水平：
  - update 40: `success≈0.4`
  - update 60: `success≈0.4`
  - update 80: `success≈0.4`
- 但 final eval 回落到：
  - final `success_rate_mean=0.3`
  - final `collision_rate_mean≈0.005`
- formation DAgger BC 和 formation success-policy BC 都没有超过这条 PPO 分支

结论：

- `formation` 当前已经从“不稳定”推进到“可用且接近 heuristic 上限”
- 但还不能像 `goal_nav` / `risk_nav` 那样被判定为“完全修通”

### 8. coverage 成功轨迹强化的最新结论

- 从 `phase10` PPO checkpoint 继续收 coverage 成功轨迹，样本密度仍然偏低：
  - `u60` deterministic: `2 / 240` 成功 episode
  - `u40` deterministic: `0 / 240`
  - `u60` with `action_noise_std=0.05`: 仍只有 `2 / 240`
- 因此“少量 success-only reinforcement”当前不足以单独修通 coverage
- `phase10` 依然是当前 coverage 的最佳 PPO 主线：
  - 多个连续 eval 点 `success≈0.10`
  - `coverage_ratio≈0.67~0.68`
  - `collision≈0.00025~0.00063`
- `phase15`, `phase17`, `phase18` 这些后续轻量修正都没有超过 `phase10`

结论：

- coverage 当前已经是“局部修通但终局不稳”
- 下一步如果继续推进，最可能需要真正更强的组级协调 actor，而不是再叠一层轻量偏置

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
