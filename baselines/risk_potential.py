from __future__ import annotations

import numpy as np

from baselines.greedy_goal import _extract_peaks


class RiskAwarePotentialPolicy:
    def __init__(self, config: dict):
        self.config = config
        self.max_waypoint_step = float(config["max_waypoint_step"])

    def act(self, observation: dict) -> np.ndarray:
        positions = observation["agents"][:, :2]
        task_field = observation["task_field"]
        map_size = float(observation["global_info"][-1])
        if task_field.shape[0] == 1:
            return np.zeros((len(positions), 2), dtype=np.float32)
        goal_map = task_field[1]
        risk_map = task_field[4]
        obstacle_map = task_field[0]
        goals = _extract_peaks(goal_map, max(len(positions), 1), map_size=map_size)
        grad_y_risk, grad_x_risk = np.gradient(risk_map)
        grad_y_obs, grad_x_obs = np.gradient(obstacle_map)
        actions = []
        for pos in positions:
            target = goals[np.argmin(np.linalg.norm(goals - pos[None, :], axis=-1))] if len(goals) else np.array([0.5 * map_size, 0.5 * map_size], dtype=np.float32)
            goal_vec = target - pos
            ix = int(round(pos[0] / max(map_size, 1e-6) * (risk_map.shape[1] - 1)))
            iy = int(round(pos[1] / max(map_size, 1e-6) * (risk_map.shape[0] - 1)))
            grad = np.array(
                [
                    grad_x_risk[iy, ix] + 1.5 * grad_x_obs[iy, ix],
                    grad_y_risk[iy, ix] + 1.5 * grad_y_obs[iy, ix],
                ],
                dtype=np.float32,
            )
            direction = goal_vec - 0.35 * grad
            actions.append(np.clip(direction / max(self.max_waypoint_step, 1e-6), -1.0, 1.0))
        return np.asarray(actions, dtype=np.float32)
