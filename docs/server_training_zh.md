# Ubuntu Docker 服务器训练指南

本文面向 Ubuntu 服务器上的 Docker 训练流程。Wayffusion 是二维数值化 multi-UAV RL benchmark，不需要 Isaac Sim、ROS、PX4、真实飞控或图形桌面。服务器训练应默认 headless，Matplotlib 使用 `Agg` 后端。

## 1. 推荐 Docker 启动命令

使用服务器已验证的 PyTorch CUDA 镜像：

```bash
docker run --gpus all -it \
  --name wayffusion \
  --network=host \
  --ipc=host \
  --shm-size=16g \
  -e MPLBACKEND=Agg \
  -e http_proxy=http://127.0.0.1:7891 \
  -e https_proxy=http://127.0.0.1:7891 \
  -e HTTP_PROXY=http://127.0.0.1:7891 \
  -e HTTPS_PROXY=http://127.0.0.1:7891 \
  -v /home/zhaozihan/data0/Wayffusion:/workspace/Wayffusion \
  -w /workspace/Wayffusion \
  pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel \
  /bin/bash
```

这里使用 `--network=host`，因为容器需要访问服务器本地代理 `127.0.0.1:7891`。同时设置 `MPLBACKEND=Agg`，避免训练或录像保存时尝试打开 GUI。

如果挂载目录中仓库位于 `/workspace/Wayffusion/Wayffusion`，进入仓库根目录后再运行命令：

```bash
cd /workspace/Wayffusion/Wayffusion
```

在当前服务器上，已验证可用的 Python 解释器是：

```bash
export PYTHON_BIN=/opt/conda/bin/python
export CUDA_VISIBLE_DEVICES=0
export MPLBACKEND=Agg
```

其中 GPU 0/1/3 通常空闲，GPU 2 可能已有其他训练任务。需要换卡时只改 `CUDA_VISIBLE_DEVICES`。

## 2. 代理验证

宿主机上先确认代理监听和外网连通：

```bash
ss -ltnp | grep 7891
curl -I -x http://127.0.0.1:7891 https://github.com
```

容器里再确认代理和 Git HTTPS：

```bash
curl -I -x http://127.0.0.1:7891 https://github.com
git ls-remote https://github.com/xiancaigua/Wayffusion.git
```

## 3. 安装依赖

该 Docker 镜像已经自带 `torch 2.7.0 + CUDA 12.8`。服务器环境不要直接安装通用依赖文件：

```bash
pip install -r requirements.txt
```

原因是通用文件包含 `torch>=2.9`，会触发 pip 尝试升级 torch，破坏镜像内置 CUDA 组合。服务器应使用：

```bash
${PYTHON_BIN:-python} -m pip install -r requirements-server.txt
```

`requirements-server.txt` 只包含训练、评估、测试和可视化所需的非 torch 依赖，包括 `psutil`、`tensorboard`、`pytest`、`gymnasium`、`matplotlib`、`imageio` 等。

## 4. GPU 验证

运行：

```bash
${PYTHON_BIN:-python} - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("torch_cuda:", torch.version.cuda)
print("gpu_count:", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    print(f"gpu_{idx}:", torch.cuda.get_device_name(idx))
PY
```

也可以使用仓库脚本做更完整的环境检查：

```bash
${PYTHON_BIN:-python} scripts/check/server/check_server_env.py
```

## 5. Smoke test

先跑单元测试：

```bash
${PYTHON_BIN:-python} -m pytest tests/
```

再生成任务场和基础 rollout 图：

```bash
${PYTHON_BIN:-python} scripts/check/generate_task_fields.py --config configs/env/multitask.yaml
```

这些输出写入 `outputs/smoke/...`。

## 6. 最小训练验证

推荐先跑极短 PPO 训练，确认 checkpoint、TensorBoard 和 GIF 录像都能写入：

```bash
PYTHON_BIN=/opt/conda/bin/python CUDA_VISIBLE_DEVICES=0 bash scripts/check/server/smoke_train_ppo.sh
```

如果要一次覆盖 PPO、SAC、TD3、BC 以及测试、专家数据生成、checkpoint、metrics 和 TensorBoard 日志写出，使用：

```bash
PYTHON_BIN=/opt/conda/bin/python CUDA_VISIBLE_DEVICES=0 bash scripts/check/server/smoke_train_all.sh
```

训练输出保持项目既有结构：

```text
outputs/training/ppo/<timestamp>/<run_name>/
  checkpoints/
  snapshot/
  tensorboard/
  media/
  final_eval_media/
  training_metrics.csv
  eval_metrics.csv
```

## 7. TensorBoard

容器里启动：

```bash
tensorboard --logdir outputs/training --host 127.0.0.1 --port 6006
```

或使用脚本：

```bash
TENSORBOARD_BIN=/opt/conda/bin/tensorboard bash scripts/check/server/start_tensorboard.sh
```

本地机器建立 SSH 转发：

```bash
ssh -N -L 16006:127.0.0.1:6006 zhaozihan@服务器IP
```

浏览器打开：

```text
http://localhost:16006
```

Docker 使用 `--network=host` 时不需要 `-p 6006:6006`。

## 8. 长时间训练建议

使用 `tmux`：

```bash
tmux new -s wayffusion
python scripts/train_ppo.py \
  --config configs/policy/ppo_cnn_deepsets_multitask_20k.yaml \
  --tasks goal_nav coverage formation risk_nav \
  --agent_counts 4 \
  --total_updates 850 \
  --target_episodes 0 \
  --console_log_interval 5
```

或使用 `nohup`：

```bash
mkdir -p outputs/training
nohup python scripts/train_ppo.py \
  --config configs/policy/ppo_cnn_deepsets_multitask_20k.yaml \
  --tasks goal_nav coverage formation risk_nav \
  --agent_counts 4 \
  --total_updates 850 \
  --target_episodes 0 \
  --console_log_interval 5 \
  > outputs/training/server_train.log 2>&1 &
```

训练默认 `--headless`，不要在服务器长训中使用 `--no-headless`。

## 9. 常见问题

### git: command not found

安装 Git：

```bash
apt-get update
apt-get install -y git
```

### npm/node/codex 不相关

训练依赖不需要 npm、node 或 Codex CLI。不要把它们写入 Python 训练依赖。

### matplotlib GUI 报错

确认容器启动时设置了：

```bash
export MPLBACKEND=Agg
```

训练脚本默认 headless。服务器上不要默认使用 `--no-headless`。

### TensorBoard 访问不到

确认容器里 TensorBoard 绑定到服务器本地地址：

```bash
tensorboard --logdir outputs/training --host 127.0.0.1 --port 6006
```

再从本地机器做 SSH 转发：

```bash
ssh -N -L 16006:127.0.0.1:6006 zhaozihan@服务器IP
```

浏览器访问 `http://localhost:16006`。使用 `--network=host` 时不需要 `-p 6006:6006`。

### torch 被 pip 升级

服务器镜像已有 `torch 2.7.0 + CUDA 12.8`。只运行：

```bash
pip install -r requirements-server.txt
```

不要在服务器容器里直接安装 `requirements.txt`。

### psutil 缺失

`utils/profiling.py` 需要 `psutil`。服务器依赖文件已经包含它：

```bash
pip install -r requirements-server.txt
```

### 容器名 wayffusion 冲突

如果旧容器还在，先查看：

```bash
docker ps -a | grep wayffusion
```

删除不再使用的旧容器：

```bash
docker rm wayffusion
```

或者启动时换一个容器名。
