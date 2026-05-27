from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

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
            "last_goal_coverage_ratio": 0.0,
            "success_bonus_paid": False,
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
        previous_goal_cost = float(task_state["goal_progress"])
        goal_cost = self._goal_cost(task_state["goals"], env_state["positions"])
        progress = previous_goal_cost - goal_cost
        task_state["goal_progress"] = goal_cost

        distances = np.linalg.norm(
            env_state["positions"][:, None, :] - task_state["goals"][None, :, :], axis=-1
        )
        goal_radius = self.config["goal_radius"] * env_state.get("spatial_scale", 1.0)
        newly_reached = (~task_state["goal_reached"]) & (distances.min(axis=0) <= goal_radius)
        task_state["goal_reached"] |= newly_reached
        coverage_ratio = float(task_state["goal_reached"].mean()) if len(task_state["goal_reached"]) else 0.0
        previous_coverage_ratio = float(task_state.get("last_goal_coverage_ratio", 0.0))
        coverage_delta = max(coverage_ratio - previous_coverage_ratio, 0.0)
        task_state["last_goal_coverage_ratio"] = coverage_ratio

        repeated_goal = 0
        if task_state["goal_reached"].any():
            reached_distances = distances[:, task_state["goal_reached"]]
            per_goal_counts = (reached_distances <= goal_radius).sum(axis=0)
            repeated_goal = int(np.maximum(per_goal_counts - 1, 0).sum())
        repeated_ratio = float(repeated_goal) / float(max(env_state["num_agents"], 1))
        mean_goal_distance = float(distances.min(axis=0).mean()) if len(task_state["goals"]) else 0.0
        normalized_distance = mean_goal_distance / max(float(env_state.get("map_size", 1.0)), 1e-6)
        normalized_progress = float(np.clip(progress / max(goal_radius, 1e-6), -2.0, 2.0))
        mean_risk_exposure = float(transition_info["step_risk_exposure"]) / float(max(env_state["num_agents"], 1))

        weights = self.config["reward_weights"]["risk_nav"]
        metrics = self.get_metrics(task_state, env_state)
        success_bonus = 0.0
        if bool(metrics["success"]) and not bool(task_state.get("success_bonus_paid", False)):
            success_bonus = float(weights.get("success_bonus", 0.0))
            task_state["success_bonus_paid"] = True

        progress_reward = float(weights["progress"] * normalized_progress)
        completion_reward = float(weights["goal_reached"] * coverage_delta)
        coverage_reward = float(weights.get("coverage", 0.0) * coverage_ratio)
        distance_penalty = float(weights.get("distance", 0.0) * normalized_distance)
        risk_task_penalty = float(weights["risk_exposure"] * mean_risk_exposure)
        repeated_penalty = float(weights.get("repeated_goal", 0.0) * repeated_ratio)
        reward = (
            progress_reward
            + completion_reward
            + coverage_reward
            + distance_penalty
            + risk_task_penalty
            + repeated_penalty
            + success_bonus
        )
        return TaskStepResult(
            reward=reward,
            success=bool(metrics["success"]),
            metrics=metrics,
            components={
                "task_progress_reward": progress_reward,
                "task_completion_reward": completion_reward,
                "task_coverage_reward": coverage_reward,
                "task_distance_penalty": distance_penalty,
                "risk_task_penalty": risk_task_penalty,
                "task_repeated_penalty": repeated_penalty,
                "task_success_bonus": success_bonus,
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
