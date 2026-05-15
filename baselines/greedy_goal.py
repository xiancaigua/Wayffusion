from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _extract_peaks(value_map: np.ndarray, count: int, map_size: float = 1.0, suppression_radius: int = 4) -> np.ndarray:
    work = value_map.copy()
    peaks = []
    for _ in range(max(count, 1)):
        flat_idx = int(np.argmax(work))
        y, x = np.unravel_index(flat_idx, work.shape)
        if work[y, x] <= 1e-6:
            break
        peaks.append(
            [
                (x / max(work.shape[1] - 1, 1)) * map_size,
                (y / max(work.shape[0] - 1, 1)) * map_size,
            ]
        )
        y0 = max(0, y - suppression_radius)
        y1 = min(work.shape[0], y + suppression_radius + 1)
        x0 = max(0, x - suppression_radius)
        x1 = min(work.shape[1], x + suppression_radius + 1)
        work[y0:y1, x0:x1] = 0.0
    return np.asarray(peaks, dtype=np.float32)


class GreedyGoalPolicy:
    def __init__(self, config: dict):
        self.config = config
        self.max_waypoint_step = float(config["max_waypoint_step"])

    def act(self, observation: dict) -> np.ndarray:
        positions = observation["agents"][:, :2]
        task_field = observation["task_field"]
        map_size = float(observation["global_info"][-1])
        if task_field.shape[0] == 1:
            value_map = task_field[0]
        else:
            value_map = task_field[1]
        goals = _extract_peaks(value_map, max(len(positions), 2), map_size=map_size)
        if len(goals) == 0:
            return np.zeros((len(positions), 2), dtype=np.float32)
        distances = np.linalg.norm(positions[:, None, :] - goals[None, :, :], axis=-1)
        assign_cols = self._assign(distances)
        targets = goals[assign_cols]
        return np.clip((targets - positions) / max(self.max_waypoint_step, 1e-6), -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _assign(distances: np.ndarray) -> np.ndarray:
        num_agents, num_goals = distances.shape
        if num_goals >= num_agents:
            rows, cols = linear_sum_assignment(distances)
            ordered = np.zeros((num_agents,), dtype=np.int32)
            ordered[rows] = cols
            return ordered
        nearest = np.argmin(distances, axis=1)
        return nearest.astype(np.int32)
