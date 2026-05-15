from __future__ import annotations

import numpy as np

from fields.field_utils import world_to_grid


class GreedyCoveragePolicy:
    def __init__(self, config: dict):
        self.config = config
        self.max_waypoint_step = float(config["max_waypoint_step"])

    def act(self, observation: dict) -> np.ndarray:
        positions = observation["agents"][:, :2]
        task_field = observation["task_field"]
        map_size = float(observation["global_info"][-1])
        if task_field.shape[0] == 1:
            score_map = task_field[0]
            visited = np.zeros_like(score_map)
            risk = np.zeros_like(score_map)
            density = np.zeros_like(score_map)
        else:
            score_map = task_field[2]
            visited = task_field[5]
            risk = task_field[4]
            density = task_field[6]
        utility = score_map * (1.0 - visited) - 0.25 * density - 0.2 * risk
        flat_indices = np.argpartition(utility.reshape(-1), -256)[-256:]
        candidates = np.stack(np.unravel_index(flat_indices, utility.shape), axis=-1)
        candidate_points = np.stack(
            [
                candidates[:, 1] / max(utility.shape[1] - 1, 1) * map_size,
                candidates[:, 0] / max(utility.shape[0] - 1, 1) * map_size,
            ],
            axis=-1,
        ).astype(np.float32)
        actions = []
        selected_targets: list[np.ndarray] = []
        for position in positions:
            scores = []
            for point in candidate_points:
                grid_idx = world_to_grid(point[None, :], utility.shape[0], map_size=map_size)[0]
                base = float(utility[grid_idx[1], grid_idx[0]])
                dist_bonus = 0.1 * float(np.linalg.norm(point - position))
                crowding_penalty = 0.0
                if selected_targets:
                    crowding_penalty = sum(0.08 / max(np.linalg.norm(point - other), 0.05) for other in selected_targets)
                scores.append(base + dist_bonus - crowding_penalty)
            target = candidate_points[int(np.argmax(scores))] if len(candidate_points) else position
            selected_targets.append(target)
            actions.append(np.clip((target - position) / max(self.max_waypoint_step, 1e-6), -1.0, 1.0))
        return np.asarray(actions, dtype=np.float32)
