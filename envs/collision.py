from __future__ import annotations

import numpy as np

from fields.field_utils import world_to_grid


def obstacle_collision_mask(positions: np.ndarray, obstacle_map: np.ndarray, map_size: float = 1.0) -> np.ndarray:
    if len(positions) == 0:
        return np.zeros((0,), dtype=bool)
    indices = world_to_grid(positions, obstacle_map.shape[0], map_size=map_size)
    return obstacle_map[indices[:, 1], indices[:, 0]] > 0.5


def pairwise_collision_pairs(positions: np.ndarray, collision_radius: float) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if np.linalg.norm(positions[i] - positions[j]) < collision_radius:
                pairs.append((i, j))
    return pairs
