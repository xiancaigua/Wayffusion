from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from fields.field_utils import gaussian_map, normalize_map, sample_points
from tasks.base_task import BaseTask, TaskStepResult


class FormationTask(BaseTask):
    name = "formation"
    task_id = 2

    def _radius_penalty_weight(self) -> float:
        weights = self.config["reward_weights"]["formation"]
        if "radius_error_penalty" in weights:
            return float(abs(weights["radius_error_penalty"]))
        return float(abs(weights.get("radius_error", 0.0)))

    def reset(self, rng: np.random.Generator, env_state: dict) -> dict:
        templates = self.config["formation"]["train_templates"]
        template = str(rng.choice(templates))
        target = sample_points(rng, 1, margin=0.2, map_size=env_state["map_size"])[0]
        velocity = np.zeros(2, dtype=np.float32)
        if self.config.get("target_motion", "static") == "linear":
            velocity = rng.uniform(
                -0.012 * env_state["spatial_scale"], 0.012 * env_state["spatial_scale"], size=(2,)
            ).astype(np.float32)
        radius = float(self.config["formation_radius"]) * env_state["spatial_scale"]
        slots = self._template_slots(template, target, env_state["num_agents"], radius, env_state["map_size"])
        return {
            "template": template,
            "target_position": target.astype(np.float32),
            "target_velocity": velocity,
            "radius": radius,
            "slots": slots,
            "last_error": self._formation_error(env_state["positions"], slots),
            "success_bonus_paid": False,
        }

    def step_update(self, task_state, env_state) -> None:
        target = task_state["target_position"] + task_state["target_velocity"] * env_state["dt"]
        lower = 0.15 * env_state["map_size"]
        upper = 0.85 * env_state["map_size"]
        if np.any((target < lower) | (target > upper)):
            task_state["target_velocity"] *= -1.0
            target = np.clip(task_state["target_position"] + task_state["target_velocity"] * env_state["dt"], lower, upper)
        task_state["target_position"] = target.astype(np.float32)
        task_state["slots"] = self._template_slots(
            task_state["template"],
            task_state["target_position"],
            env_state["num_agents"],
            task_state["radius"],
            env_state["map_size"],
        )

    def build_field(self, task_state, env_state) -> dict:
        target_map = gaussian_map(
            task_state["target_position"][None, :],
            env_state["grid_size"],
            sigma=0.04 * env_state["spatial_scale"],
            map_size=env_state["map_size"],
        )
        desired_occ = gaussian_map(
            task_state["slots"],
            env_state["grid_size"],
            sigma=0.04 * env_state["spatial_scale"],
            map_size=env_state["map_size"],
        )
        template_map = normalize_map(
            gaussian_map(
                task_state["slots"],
                env_state["grid_size"],
                sigma=0.08 * env_state["spatial_scale"],
                map_size=env_state["map_size"],
            )
        )
        return {
            "goal_reward": target_map,
            "desired_occupancy": desired_occ,
            "formation_template": template_map,
        }

    def compute_reward(self, task_state, prev_env_state, env_state, transition_info) -> TaskStepResult:
        error = self._formation_error(env_state["positions"], task_state["slots"])
        progress = float(task_state["last_error"] - error)
        task_state["last_error"] = error

        angular_uniformity = self._angular_uniformity(env_state["positions"], task_state["target_position"])
        radius_error = self._radius_error(env_state["positions"], task_state["target_position"], task_state["radius"])
        stability = 1.0 / (1.0 + float(np.std(env_state["formation_error_history"][-5:] or [error])))
        scale = max(float(task_state["radius"]), 1e-6)
        max_step_distance = max(float(transition_info.get("max_step_distance", scale)), 1e-6)
        normalized_progress = float(np.clip(progress / max_step_distance, -5.0, 5.0))
        normalized_error = float(error / scale)
        normalized_radius_error = float(radius_error / scale)
        proximity = float(1.0 / (1.0 + normalized_error + normalized_radius_error))

        weights = self.config["reward_weights"]["formation"]
        radius_penalty_weight = self._radius_penalty_weight()
        metrics = self.get_metrics(task_state, env_state)
        success_bonus = 0.0
        if bool(metrics["success"]) and not bool(task_state.get("success_bonus_paid", False)):
            success_bonus = float(weights.get("success_bonus", 0.0))
            task_state["success_bonus_paid"] = True

        formation_reward = float(weights["error_reduction"] * normalized_progress)
        error_penalty = float(-abs(weights.get("error_penalty", 0.0)) * normalized_error)
        angular_reward = float(weights["angular_coverage"] * angular_uniformity)
        radius_penalty = float(-radius_penalty_weight * normalized_radius_error)
        stability_reward = float(weights.get("stability", 0.0) * stability)
        proximity_reward = float(weights.get("slot_proximity", 0.0) * proximity)
        reward = formation_reward + error_penalty + angular_reward + radius_penalty + stability_reward + proximity_reward + success_bonus
        return TaskStepResult(
            reward=reward,
            success=bool(metrics["success"]),
            metrics=metrics,
            components={
                "formation_reward": formation_reward,
                "formation_error_penalty": error_penalty,
                "angular_reward": angular_reward,
                "radius_penalty": radius_penalty,
                "stability_reward": stability_reward,
                "slot_proximity_reward": proximity_reward,
                "formation_success_bonus": success_bonus,
            },
        )

    def get_metrics(self, task_state, env_state) -> dict:
        error = self._formation_error(env_state["positions"], task_state["slots"])
        angular_uniformity = self._angular_uniformity(env_state["positions"], task_state["target_position"])
        radius_error = self._radius_error(env_state["positions"], task_state["target_position"], task_state["radius"])
        stability = 1.0 / (1.0 + float(np.std(env_state["formation_error_history"][-10:] or [error])))
        tolerance = self.config["formation_tolerance"] * env_state["spatial_scale"]
        success = (
            error <= tolerance
            and radius_error <= tolerance
            and angular_uniformity >= 1.0 - self.config["formation"]["angular_tolerance"]
        )
        return {
            "success": float(success),
            "formation_error": error,
            "angular_coverage_uniformity": angular_uniformity,
            "radius_error": radius_error,
            "formation_stability": stability,
        }

    @staticmethod
    def _template_slots(template: str, center: np.ndarray, count: int, radius: float, map_size: float) -> np.ndarray:
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False, dtype=np.float32)
        if template == "line":
            offsets = np.stack(
                [np.linspace(-radius, radius, count, dtype=np.float32), np.zeros(count, dtype=np.float32)],
                axis=-1,
            )
        elif template == "diamond":
            base = np.array([[0, radius], [radius, 0], [0, -radius], [-radius, 0]], dtype=np.float32)
            repeats = int(np.ceil(count / len(base)))
            offsets = np.concatenate([base for _ in range(repeats)], axis=0)[:count]
        elif template == "arc":
            arc_angles = np.linspace(-0.85, 0.85, count, dtype=np.float32) * np.pi
            offsets = np.stack([np.cos(arc_angles), np.sin(arc_angles)], axis=-1) * radius
        else:
            offsets = np.stack([np.cos(angles), np.sin(angles)], axis=-1) * radius
        lower = 0.05 * map_size
        upper = 0.95 * map_size
        return np.clip(center[None, :] + offsets, lower, upper).astype(np.float32)

    @staticmethod
    def _formation_error(positions: np.ndarray, slots: np.ndarray) -> float:
        distances = np.linalg.norm(positions[:, None, :] - slots[None, :, :], axis=-1)
        rows, cols = linear_sum_assignment(distances)
        return float(distances[rows, cols].mean())

    @staticmethod
    def _angular_uniformity(positions: np.ndarray, center: np.ndarray) -> float:
        rel = positions - center[None, :]
        angles = np.sort(np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2 * np.pi))
        gaps = np.diff(np.concatenate([angles, angles[:1] + 2 * np.pi]))
        target_gap = 2 * np.pi / max(len(positions), 1)
        return float(np.clip(1.0 - np.abs(gaps - target_gap).mean() / max(target_gap, 1e-6), 0.0, 1.0))

    @staticmethod
    def _radius_error(positions: np.ndarray, center: np.ndarray, radius: float) -> float:
        dists = np.linalg.norm(positions - center[None, :], axis=-1)
        return float(np.abs(dists - radius).mean())
