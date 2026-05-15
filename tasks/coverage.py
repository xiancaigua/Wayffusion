from __future__ import annotations

import numpy as np

from fields.field_utils import gaussian_map, normalize_map, sample_points
from tasks.base_task import BaseTask, TaskStepResult


class CoverageTask(BaseTask):
    name = "coverage"
    task_id = 1

    def reset(self, rng: np.random.Generator, env_state: dict) -> dict:
        multi_peak = bool(self.config["coverage"]["multi_peak_probability"])
        task_count_scale = env_state.get("task_count_scale", 1.0)
        scaling_mode = str(env_state.get("scaling_mode", "fixed_map"))
        density_peak_scale_exponent = float(self.config["coverage"].get("density_peak_scale_exponent", 0.8))
        peak_scale = task_count_scale if scaling_mode == "fixed_map" else task_count_scale ** density_peak_scale_exponent
        peak_count = (5 if multi_peak else 2) + max(0, int(round(peak_scale)) - 1)
        required_visits = 1
        if scaling_mode == "fixed_map":
            required_visits = max(1, int(np.ceil(np.sqrt(task_count_scale))))
        else:
            required_visits = max(1, int(np.ceil(task_count_scale ** 0.25)))
        centers = sample_points(rng, peak_count, margin=0.08, map_size=env_state["map_size"])
        amplitudes = rng.uniform(0.5, 1.0, size=(peak_count,)).astype(np.float32)
        target_probability = normalize_map(
            gaussian_map(
                centers,
                env_state["grid_size"],
                sigma=0.09 * env_state["spatial_scale"],
                amplitudes=amplitudes,
                map_size=env_state["map_size"],
            )
        )
        coverage_demand = (target_probability > np.quantile(target_probability, 0.55)).astype(np.float32)
        return {
            "target_probability": target_probability,
            "coverage_demand": coverage_demand,
            "required_visits": required_visits,
            "last_visited_score": 0.0,
            "total_repeated": 0.0,
            "total_detected": 0.0,
        }

    def build_field(self, task_state, env_state) -> dict:
        return {
            "target_probability": task_state["target_probability"],
            "desired_occupancy": np.clip(
                task_state["coverage_demand"] * min(float(task_state["required_visits"]) / 4.0, 1.0),
                0.0,
                1.0,
            ),
        }

    def compute_reward(self, task_state, prev_env_state, env_state, transition_info) -> TaskStepResult:
        required_visits = max(int(task_state.get("required_visits", 1)), 1)
        prev_fulfillment = np.clip(prev_env_state["visit_count_map"] / required_visits, 0.0, 1.0)
        current_fulfillment = np.clip(env_state["visit_count_map"] / required_visits, 0.0, 1.0)
        new_mask = np.clip(current_fulfillment - prev_fulfillment, 0.0, 1.0)
        repeated_mask = (prev_env_state["visit_count_map"] >= required_visits).astype(np.float32) * env_state["step_coverage_mask"]

        new_coverage = float((new_mask * task_state["coverage_demand"]).sum() / np.maximum(task_state["coverage_demand"].sum(), 1.0))
        detection_gain = float(
            (new_mask * task_state["target_probability"]).sum()
            / np.maximum(task_state["target_probability"].sum(), 1e-6)
        )
        repeated_ratio = float(repeated_mask.sum() / np.maximum(env_state["step_coverage_mask"].sum(), 1.0))

        task_state["total_repeated"] += repeated_mask.sum()
        task_state["total_detected"] += detection_gain

        weights = self.config["reward_weights"]["coverage"]
        reward = (
            weights["new_coverage"] * new_coverage
            + weights["high_probability"] * detection_gain
            + weights["repeated_coverage"] * repeated_ratio
        )
        metrics = self.get_metrics(task_state, env_state)
        return TaskStepResult(
            reward=reward,
            success=bool(metrics["success"]),
            metrics=metrics,
            components={
                "coverage_reward": weights["new_coverage"] * new_coverage,
                "detection_reward": weights["high_probability"] * detection_gain,
                "repeated_coverage_penalty": weights["repeated_coverage"] * repeated_ratio,
            },
        )

    def get_metrics(self, task_state, env_state) -> dict:
        demand = np.maximum(task_state["coverage_demand"].sum(), 1.0)
        required_visits = max(int(task_state.get("required_visits", 1)), 1)
        fulfillment_map = np.clip(env_state["visit_count_map"] / required_visits, 0.0, 1.0)
        coverage_ratio = float((fulfillment_map * task_state["coverage_demand"]).sum() / demand)
        detection_score = float(
            (fulfillment_map * task_state["target_probability"]).sum()
            / np.maximum(task_state["target_probability"].sum(), 1e-6)
        )
        visit_counts = env_state["visit_count_map"]
        repeated_ratio = float((visit_counts > required_visits).sum() / np.maximum((visit_counts > 0).sum(), 1.0))
        positions = env_state["positions"]
        centroid = positions.mean(axis=0, keepdims=True)
        dispersion = float(np.linalg.norm(positions - centroid, axis=-1).mean())
        min_steps = max(1, int(np.ceil(2.0 * np.sqrt(max(env_state.get("task_count_scale", 1.0), 1.0)))))
        success_ratio = float(self.config["coverage"]["success_ratio"]) + 0.05 * max(required_visits - 1, 0)
        success_ratio = min(success_ratio, 0.999)
        success = coverage_ratio >= success_ratio and env_state["step_count"] >= min_steps
        return {
            "success": float(success),
            "coverage_ratio": coverage_ratio,
            "accumulated_detection_probability": detection_score,
            "repeated_coverage_ratio": repeated_ratio,
            "time_discounted_detection_score": detection_score / (1.0 + 0.02 * env_state["step_count"]),
            "spatial_dispersion": dispersion,
        }
