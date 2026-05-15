from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from fields.field_utils import gaussian_map, sample_points
from tasks.base_task import BaseTask, TaskStepResult


class GoalNavigationTask(BaseTask):
    name = "goal_nav"
    task_id = 0

    def reset(self, rng: np.random.Generator, env_state: dict) -> dict:
        num_agents = env_state["num_agents"]
        goal_low, goal_high = self.config["goal_nav"]["num_goals_range"]
        task_count_scale = env_state.get("task_count_scale", 1.0)
        scaling_mode = str(env_state.get("scaling_mode", "fixed_map"))
        density_goal_scale_exponent = float(self.config["goal_nav"].get("density_goal_scale_exponent", 0.8))
        goal_scale = task_count_scale if scaling_mode == "fixed_map" else task_count_scale ** density_goal_scale_exponent
        goal_low = max(1, int(round(goal_low * goal_scale)))
        goal_high = max(goal_low, int(round(goal_high * goal_scale)))
        num_goals = int(rng.integers(goal_low, goal_high + 1))
        goals = sample_points(rng, num_goals, margin=0.12, map_size=env_state["map_size"])
        return {
            "goals": goals,
            "goal_reached": np.zeros((num_goals,), dtype=bool),
            "goal_progress": self._goal_cost(goals, env_state["positions"]),
            "goal_reward_map": gaussian_map(
                goals,
                env_state["grid_size"],
                sigma=0.055 * env_state["spatial_scale"],
                map_size=env_state["map_size"],
            ),
            "num_agents": num_agents,
        }

    def build_field(self, task_state: dict, env_state: dict) -> dict:
        return {"goal_reward": task_state["goal_reward_map"]}

    def compute_reward(self, task_state, prev_env_state, env_state, transition_info) -> TaskStepResult:
        goal_cost = self._goal_cost(task_state["goals"], env_state["positions"])
        progress = task_state["goal_progress"] - goal_cost
        task_state["goal_progress"] = goal_cost

        goal_radius = self.config["goal_radius"] * env_state.get("spatial_scale", 1.0)
        distances = np.linalg.norm(
            env_state["positions"][:, None, :] - task_state["goals"][None, :, :], axis=-1
        )
        newly_reached = (~task_state["goal_reached"]) & (distances.min(axis=0) <= goal_radius)
        task_state["goal_reached"] |= newly_reached

        repeated_goal = 0
        if task_state["goal_reached"].any():
            repeated_goal = int((distances[:, task_state["goal_reached"]] <= goal_radius).sum())
        goal_count = max(len(task_state["goals"]), 1)
        completion_ratio = float(newly_reached.sum()) / float(goal_count)
        repeated_ratio = float(repeated_goal) / float(max(env_state["num_agents"], 1))

        weights = self.config["reward_weights"]["goal_nav"]
        reward = (
            weights["progress"] * float(progress)
            + weights["goal_reached"] * completion_ratio
            + weights["repeated_goal"] * repeated_ratio
        )
        metrics = self.get_metrics(task_state, env_state)
        return TaskStepResult(
            reward=reward,
            success=bool(metrics["success"]),
            metrics=metrics,
            components={
                "task_progress_reward": weights["progress"] * float(progress),
                "task_completion_reward": weights["goal_reached"] * completion_ratio,
                "task_repeated_penalty": weights["repeated_goal"] * repeated_ratio,
            },
        )

    def get_metrics(self, task_state, env_state) -> dict:
        coverage_ratio = float(task_state["goal_reached"].mean()) if len(task_state["goal_reached"]) else 0.0
        success_ratio = float(self.config["goal_nav"].get("success_ratio", 0.995))
        min_steps = max(1, int(np.ceil(2.0 * np.sqrt(max(env_state.get("task_count_scale", 1.0), 1.0)))))
        success = coverage_ratio >= success_ratio and env_state["step_count"] >= min_steps
        return {
            "success": float(success),
            "goal_coverage_ratio": coverage_ratio,
            "remaining_goal_ratio": 1.0 - coverage_ratio,
            "completion_time": float(env_state["step_count"]),
        }

    @staticmethod
    def _goal_cost(goals: np.ndarray, positions: np.ndarray) -> float:
        if len(goals) == 0 or len(positions) == 0:
            return 0.0
        distances = np.linalg.norm(positions[:, None, :] - goals[None, :, :], axis=-1)
        if distances.shape[0] <= distances.shape[1]:
            rows, cols = linear_sum_assignment(distances)
            assigned = distances[rows, cols].sum()
            if distances.shape[1] > distances.shape[0]:
                assigned += distances.min(axis=0).sum() - distances[:, cols].min(axis=0).sum()
            denom = max(max(distances.shape[0], distances.shape[1]), 1)
            return float(assigned / denom)
        rows, cols = linear_sum_assignment(distances.T)
        denom = max(max(distances.shape[0], distances.shape[1]), 1)
        return float(distances.T[rows, cols].sum() / denom)
