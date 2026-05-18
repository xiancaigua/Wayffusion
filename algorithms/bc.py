from __future__ import annotations

from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from policies import observation_to_tensor
from utils.profiling import get_memory_usage_mb


class BCTrainer:
    def __init__(self, policy: nn.Module, train_config: dict, device: str | None = None):
        self.policy = policy
        self.train_config = train_config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(train_config["learning_rate"]))

    def _masked_mse(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        loss = (prediction - target) ** 2
        if mask is not None:
            loss = loss * mask.unsqueeze(-1).float()
            denom = torch.clamp(mask.sum() * prediction.shape[-1], min=1.0)
            return loss.sum() / denom
        return loss.mean()

    def train(
        self,
        dataset,
        output_dir: str | Path,
        eval_env=None,
        log_callback: Callable[[dict], None] | None = None,
    ) -> list[dict]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        loader = DataLoader(dataset, batch_size=int(self.train_config["batch_size"]), shuffle=True)
        history = []
        train_start = time.perf_counter()
        for epoch in range(1, int(self.train_config["epochs"]) + 1):
            epoch_start = time.perf_counter()
            losses = []
            for batch in loader:
                obs = {
                    "task_field": batch["task_field"].to(self.device),
                    "agents": batch["agents"].to(self.device),
                    "task_id": batch["task_id"].to(self.device),
                    "global_info": batch["global_info"].to(self.device),
                    "agent_mask": batch["agent_mask"].to(self.device),
                }
                target_action = batch["action"].to(self.device)
                pred_action = self.policy.act_deterministic(obs) if hasattr(self.policy, "act_deterministic") else self.policy(obs)[0]
                loss = self._masked_mse(pred_action, target_action, obs["agent_mask"])
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()
                losses.append(float(loss.item()))
            record = {
                "epoch": epoch,
                "bc_loss": float(np.mean(losses)),
                "epoch_time_sec": float(time.perf_counter() - epoch_start),
                "wall_clock_time": float(time.perf_counter() - train_start),
                "memory_usage_mb": get_memory_usage_mb(),
            }
            checkpoint_path = checkpoints_dir / f"checkpoint_{epoch:04d}.pt"
            torch.save({"model_state_dict": self.policy.state_dict(), "train_config": self.train_config}, checkpoint_path)
            record["checkpoint_path"] = str(checkpoint_path)
            history.append(record)
            if log_callback is not None:
                log_callback(record)
        return history
