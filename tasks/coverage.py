from __future__ import annotations

import numpy as np

from fields.field_utils import gaussian_map, normalize_map, sample_points
from tasks.base_task import BaseTask, TaskStepResult


class CoverageTask(BaseTask):
    name = "coverage"
    task_id = 1

    def _success_ratio(self, task_state, env_state) -> float:
        required_visits = max(int(task_state.get("required_visits", 1)), 1)
        success_ratio = float(self.config["coverage"]["success_ratio"]) + 0.05 * max(required_visits - 1, 0)
        return min(success_ratio, 0.999)

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
            "last_coverage_ratio": 0.0,
            "last_detection_score": 0.0,
            "success_bonus_paid": False,
            "paid_milestones": set(),
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

        metrics = self.get_metrics(task_state, env_state)
        previous_coverage_ratio = float(task_state.get("last_coverage_ratio", 0.0))
        previous_detection_score = float(task_state.get("last_detection_score", 0.0))
        coverage_delta = max(float(metrics["coverage_ratio"]) - previous_coverage_ratio, 0.0)
        detection_delta = max(float(metrics["accumulated_detection_probability"]) - previous_detection_score, 0.0)
        task_state["last_coverage_ratio"] = float(metrics["coverage_ratio"])
        task_state["last_detection_score"] = float(metrics["accumulated_detection_probability"])

        weights = self.config["reward_weights"]["coverage"]
        success_bonus = 0.0
        if bool(metrics["success"]) and not bool(task_state.get("success_bonus_paid", False)):
            success_bonus = float(weights.get("success_bonus", 0.0))
            task_state["success_bonus_paid"] = True

        coverage_reward = float(weights["new_coverage"] * max(new_coverage, coverage_delta))
        detection_reward = float(weights["high_probability"] * max(detection_gain, detection_delta))
        coverage_level_reward = float(weights.get("coverage_level", 0.0) * float(metrics["coverage_ratio"]))
        milestone_reward = self._milestone_reward(task_state, float(metrics["coverage_ratio"]), weights)
        success_ratio = self._success_ratio(task_state, env_state)
        shortfall = max(success_ratio - float(metrics["coverage_ratio"]), 0.0)
        shortfall_penalty = float(weights.get("coverage_shortfall", 0.0) * shortfall)
        failure_penalty = 0.0
        if not bool(metrics["success"]) and env_state["step_count"] >= int(env_state["max_steps"]):
            failure_penalty = float(weights.get("failure_penalty", 0.0))
        repeated_penalty = float(weights["repeated_coverage"] * repeated_ratio)
        terminal_repeated_penalty = 0.0
        if env_state["step_count"] >= int(env_state["max_steps"]):
            terminal_repeated_penalty = float(weights.get("terminal_repeated_coverage", 0.0) * float(metrics["repeated_coverage_ratio"]))
        reward = (
            coverage_reward
            + detection_reward
            + coverage_level_reward
            + milestone_reward
            + shortfall_penalty
            + failure_penalty
            + repeated_penalty
            + terminal_repeated_penalty
            + success_bonus
        )
        return TaskStepResult(
            reward=reward,
            success=bool(metrics["success"]),
            metrics=metrics,
            components={
                "coverage_reward": coverage_reward,
                "detection_reward": detection_reward,
                "coverage_level_reward": coverage_level_reward,
                "coverage_milestone_reward": milestone_reward,
                "coverage_shortfall_penalty": shortfall_penalty,
                "coverage_failure_penalty": failure_penalty,
                "repeated_coverage_penalty": repeated_penalty,
                "terminal_repeated_coverage_penalty": terminal_repeated_penalty,
                "coverage_success_bonus": success_bonus,
            },
        )

    def _milestone_reward(self, task_state: dict, coverage_ratio: float, weights: dict) -> float:
        thresholds = weights.get("milestone_thresholds", [])
        bonuses = weights.get("milestone_bonuses", [])
        if not thresholds or not bonuses:
            return 0.0
        paid = task_state.setdefault("paid_milestones", set())
        reward = 0.0
        for index, threshold in enumerate(thresholds):
            threshold_value = float(threshold)
            if coverage_ratio < threshold_value or threshold_value in paid:
                continue
            bonus_idx = min(index, len(bonuses) - 1)
            reward += float(bonuses[bonus_idx])
            paid.add(threshold_value)
        return float(reward)

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
        success_ratio = self._success_ratio(task_state, env_state)
        success = coverage_ratio >= success_ratio and env_state["step_count"] >= min_steps
        return {
            "success": float(success),
            "coverage_ratio": coverage_ratio,
            "accumulated_detection_probability": detection_score,
            "repeated_coverage_ratio": repeated_ratio,
            "time_discounted_detection_score": detection_score / (1.0 + 0.02 * env_state["step_count"]),
            "spatial_dispersion": dispersion,
        }
