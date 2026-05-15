from __future__ import annotations

import numpy as np

from fields.field_utils import gaussian_map, sample_points
from tasks.base_task import BaseTask, TaskStepResult


class RiskAwareNavigationTask(BaseTask):
    name = "risk_nav"
    task_id = 3

    def reset(self, rng: np.random.Generator, env_state: dict) -> dict:
        goal_low, goal_high = self.config["risk_nav"]["num_goals_range"]
        task_count_scale = env_state.get("task_count_scale", 1.0)
        scaling_mode = str(env_state.get("scaling_mode", "fixed_map"))
        density_goal_scale_exponent = float(self.config["risk_nav"].get("density_goal_scale_exponent", 0.8))
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
                sigma=0.05 * env_state["spatial_scale"],
                map_size=env_state["map_size"],
            ),
        }

    def build_field(self, task_state, env_state) -> dict:
        return {"goal_reward": task_state["goal_reward_map"]}

    def compute_reward(self, task_state, prev_env_state, env_state, transition_info) -> TaskStepResult:
        goal_cost = self._goal_cost(task_state["goals"], env_state["positions"])
        progress = task_state["goal_progress"] - goal_cost
        task_state["goal_progress"] = goal_cost

        distances = np.linalg.norm(
            env_state["positions"][:, None, :] - task_state["goals"][None, :, :], axis=-1
        )
        newly_reached = (~task_state["goal_reached"]) & (
            distances.min(axis=0) <= self.config["goal_radius"] * env_state.get("spatial_scale", 1.0)
        )
        task_state["goal_reached"] |= newly_reached
        goal_count = max(len(task_state["goals"]), 1)
        completion_ratio = float(newly_reached.sum()) / float(goal_count)
        mean_risk_exposure = float(transition_info["step_risk_exposure"]) / float(max(env_state["num_agents"], 1))

        weights = self.config["reward_weights"]["risk_nav"]
        reward = (
            weights["progress"] * float(progress)
            + weights["goal_reached"] * completion_ratio
            + weights["risk_exposure"] * mean_risk_exposure
        )
        metrics = self.get_metrics(task_state, env_state)
        return TaskStepResult(
            reward=reward,
            success=bool(metrics["success"]),
            metrics=metrics,
            components={
                "task_progress_reward": weights["progress"] * float(progress),
                "task_completion_reward": weights["goal_reached"] * completion_ratio,
                "risk_task_penalty": weights["risk_exposure"] * mean_risk_exposure,
            },
        )

    def get_metrics(self, task_state, env_state) -> dict:
        coverage_ratio = float(task_state["goal_reached"].mean()) if len(task_state["goal_reached"]) else 0.0
        success_ratio = float(self.config["risk_nav"].get("success_ratio", 0.995))
        min_steps = max(1, int(np.ceil(2.0 * np.sqrt(max(env_state.get("task_count_scale", 1.0), 1.0)))))
        success = coverage_ratio >= success_ratio and env_state["step_count"] >= min_steps
        return {
            "success": float(success),
            "task_success_rate": float(success),
            "goal_coverage_ratio": coverage_ratio,
            "cumulative_risk_exposure": float(env_state["risk_exposure"] / max(env_state["num_agents"], 1)),
            "safety_violation_count": float(env_state["safety_violation_count"]),
        }

    @staticmethod
    def _goal_cost(goals: np.ndarray, positions: np.ndarray) -> float:
        if len(goals) == 0 or len(positions) == 0:
            return 0.0
        distances = np.linalg.norm(positions[:, None, :] - goals[None, :, :], axis=-1)
        return float(distances.min(axis=1).mean())
