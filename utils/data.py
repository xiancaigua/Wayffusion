from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def save_expert_dataset(output_path: str | Path, payload: dict[str, np.ndarray]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def load_expert_dataset(dataset_path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(dataset_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


@dataclass
class DatasetSpec:
    max_agents: int
    field_shape: tuple[int, int, int]


class ExpertDataset(Dataset):
    def __init__(self, dataset_path: str | Path):
        self.data = load_expert_dataset(dataset_path)
        self.length = int(self.data["task_field"].shape[0])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "task_field": torch.as_tensor(self.data["task_field"][index], dtype=torch.float32),
            "agents": torch.as_tensor(self.data["agents"][index], dtype=torch.float32),
            "task_id": torch.as_tensor(self.data["task_id"][index], dtype=torch.float32),
            "global_info": torch.as_tensor(self.data["global_info"][index], dtype=torch.float32),
            "agent_mask": torch.as_tensor(self.data["agent_mask"][index], dtype=torch.bool),
            "action": torch.as_tensor(self.data["action"][index], dtype=torch.float32),
            "num_agents": torch.as_tensor(self.data["num_agents"][index], dtype=torch.long),
            "reward": torch.as_tensor(self.data["reward"][index], dtype=torch.float32),
            "done": torch.as_tensor(self.data["done"][index], dtype=torch.float32),
        }
