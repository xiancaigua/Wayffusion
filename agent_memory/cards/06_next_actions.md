# 建议的下一步工作

## 高优先级

### 1. 刷新正式 verification 文件

建议：

- 重新生成一个和当前目录结构、测试数量、命令入口一致的 `outputs/verification.md`

原因：

- 当前文件已明显陈旧，会误导后续 agent 和用户

### 2. 统一配置命名

建议：

- 在 `ppo_mlp / mlp_ppo`
- `ppo_cnn_deepsets / cnn_deepsets_ppo`

两套命名中选一套作为主路径，并把另一套标记为 legacy 或 alias。

原因：

- 当前重复命名已经进入维护负担区间

### 3. 为评估脚本补 `--latest`

建议：

- 给 `evaluate_policy.py` 和 `evaluate_scaling.py` 增加最新 run 自动发现

原因：

- 训练目录现在有时间层，手工填 checkpoint 路径更容易出错

## 中优先级

### 4. 刷新一次 canonical PPO multitask run

建议：

- 用当前新的 `checkpoints/` 与 `snapshot/` 规范，跑一条标准多任务 PPO 训练线
- 同时保留 eval media 和 final eval artifacts

原因：

- 这样后续所有 agent 都能引用“新结构下的标准 run”

### 5. 为 agent_memory 增加自动生成脚本

建议：

- 后续可考虑增加脚本，把当前测试数、最新 run、最新 checkpoint 自动写回 manifest 或某张卡

原因：

- 当前 memory cards 是人工审计产物，后面若频繁修改，自动刷新会更稳

## 低优先级

### 6. 处理 `sitecustomize.py` 重复

建议：

- 明确是否保留双副本

### 7. 历史 outputs 分层归档

建议：

- 把旧结构 run 和新结构 run 分层标记

原因：

- 未来自动扫 `outputs/` 做比较时会更干净

## 路由建议

如果后续 agent 需要继续主线开发，建议优先路线是：

1. `verification refresh`
2. `checkpoint/latest helper`
3. `config naming cleanup`
4. `new canonical training run`
