from __future__ import annotations

from typing import Dict, Type

import numpy as np

from tasks.base_task import BaseTask
from tasks.coverage import CoverageTask
from tasks.formation import FormationTask
from tasks.goal_nav import GoalNavigationTask
from tasks.risk_nav import RiskAwareNavigationTask


TASK_REGISTRY: Dict[str, Type[BaseTask]] = {
    GoalNavigationTask.name: GoalNavigationTask,
    CoverageTask.name: CoverageTask,
    FormationTask.name: FormationTask,
    RiskAwareNavigationTask.name: RiskAwareNavigationTask,
}
TASK_ORDER = list(TASK_REGISTRY.keys())
TASK_NAME_TO_ID = {name: idx for idx, name in enumerate(TASK_ORDER)}


class TaskSampler:
    def __init__(self, config: dict):
        self.config = config
        self.tasks = {name: cls(config) for name, cls in TASK_REGISTRY.items()}
        self.available_task_names = list(config.get("task_names", TASK_ORDER))

    def sample(self, rng: np.random.Generator) -> BaseTask:
        if self.config.get("task_name"):
            return self.tasks[self.config["task_name"]]
        probs = self.config.get("task_sampling_probs", {})
        candidate_names = self.available_task_names or TASK_ORDER
        weights = np.array([float(probs.get(name, 1.0)) for name in candidate_names], dtype=np.float32)
        weights = weights / np.maximum(weights.sum(), 1e-8)
        task_name = rng.choice(candidate_names, p=weights)
        return self.tasks[str(task_name)]

    def get(self, task_name: str) -> BaseTask:
        return self.tasks[task_name]
