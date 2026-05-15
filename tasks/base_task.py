from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class TaskStepResult:
    reward: float
    success: bool
    metrics: Dict[str, float]
    components: Dict[str, float]


class BaseTask:
    name = "base"
    task_id = -1

    def __init__(self, config: dict):
        self.config = config

    def reset(self, rng, env_state: dict) -> dict:
        raise NotImplementedError

    def step_update(self, task_state: dict, env_state: dict) -> None:
        return None

    def build_field(self, task_state: dict, env_state: dict) -> dict:
        raise NotImplementedError

    def compute_reward(
        self,
        task_state: dict,
        prev_env_state: dict,
        env_state: dict,
        transition_info: dict,
    ) -> TaskStepResult:
        raise NotImplementedError

    def get_metrics(self, task_state: dict, env_state: dict) -> dict:
        raise NotImplementedError
