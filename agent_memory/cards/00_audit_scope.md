# 审计范围与方法

## 审计目标

本卡记录 2026-05-15 对 `D:\RL4UAV\AAAI` 仓库的项目级审计范围、证据来源和结论边界。目标不是重新设计 benchmark，而是为后续 agent 提供一个可信的起点：

1. 当前项目里已经存在什么。
2. 哪些内容可以直接信任和复用。
3. 哪些内容只能作为参考。
4. 哪些地方已经出现漂移、陈旧或潜在冲突。

## 审计覆盖面

本轮审计覆盖了以下资产：

- 顶层项目结构：`algorithms/`, `baselines/`, `configs/`, `docs/`, `envs/`, `fields/`, `outputs/`, `policies/`, `scripts/`, `tasks/`, `tests/`, `utils/`
- 当前训练/评估产物布局
- 近期新增的录制、快照、checkpoint、launcher、PPO 停止条件修改
- 测试用例和验证文件
- 历史结果摘要与阶段总结文件

## 证据来源

本卡和后续 memory cards 主要基于以下证据：

- 代码事实：训练脚本、算法实现、环境实现、policy 实现、公共 helper
- 配置事实：`configs/env/*`, `configs/policy/*`, `configs/eval/*`
- 文档事实：`README.md`, `docs/*.md`
- 结果事实：`outputs/eval/*.csv|*.md`, `outputs/verification.md`
- 测试事实：`tests/` 当前用例集合，以及最近一轮 `pytest` 通过状态

## 信任等级定义

- `trusted`
  - 可直接作为当前项目的默认事实来源。
  - 如果后续代码没有被再次修改，优先相信这类卡片。
- `usable_with_verification`
  - 基本可用，但引用前最好对照最新代码或最新实验再确认。
- `reference_only`
  - 只适合作为历史背景，不应用于新的结论。
- `stale_or_conflicting`
  - 已经和当前代码、目录结构或测试状态不一致。
- `missing_context`
  - 不能判断真假，或者缺少必要上下文。

## 审计边界

- 本轮审计没有重新跑完整的大规模学习实验。
- 本轮审计没有逐行重新验证所有旧结果 CSV 的来源和 seed。
- 本轮审计的“训练是否真正学起来”结论，只信任已经明确存在于 `outputs/eval/` 的摘要和当前测试，不把历史聊天中的口头结论当作最终证据。

## 路由结论

当前项目不属于空白状态，也不应该从零开始重建。最合理的后续路径是：

1. 直接复用当前 centralized benchmark 主体。
2. 把新的 `agent_memory/` 当作维护入口。
3. 对陈旧结果说明、命名漂移和评估便利性做补修。
4. 在需要新实验时，以当前训练/评估管线为主线继续推进。
