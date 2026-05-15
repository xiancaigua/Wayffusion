from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from baselines.greedy_goal import _extract_peaks


class GeometricFormationPolicy:
    def __init__(self, config: dict):
        self.config = config
        self.max_waypoint_step = float(config["max_waypoint_step"])

    def act(self, observation: dict) -> np.ndarray:
        positions = observation["agents"][:, :2]
        task_field = observation["task_field"]
        map_size = float(observation["global_info"][-1])
        if task_field.shape[0] == 1:
            desired = _extract_peaks(task_field[0], len(positions), map_size=map_size)
            target = desired.mean(axis=0) if len(desired) else np.array([0.5 * map_size, 0.5 * map_size], dtype=np.float32)
        else:
            desired = _extract_peaks(task_field[3], len(positions), map_size=map_size)
            target_peaks = _extract_peaks(task_field[1], 1, map_size=map_size)
            target = target_peaks[0] if len(target_peaks) else np.array([0.5 * map_size, 0.5 * map_size], dtype=np.float32)
        if len(desired) < len(positions):
            angles = np.linspace(0, 2 * np.pi, len(positions), endpoint=False)
            radius = float(self.config["formation_radius"]) * max(map_size, 1.0)
            desired = np.stack([np.cos(angles), np.sin(angles)], axis=-1) * radius + target[None, :]
            desired = np.clip(desired, 0.05 * map_size, 0.95 * map_size).astype(np.float32)
        cost = np.linalg.norm(positions[:, None, :] - desired[None, :, :], axis=-1)
        rows, cols = linear_sum_assignment(cost)
        assigned = desired[cols[np.argsort(rows)]]
        return np.clip((assigned - positions) / max(self.max_waypoint_step, 1e-6), -1.0, 1.0).astype(np.float32)
