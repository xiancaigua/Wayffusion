from __future__ import annotations

from typing import Dict
import warnings

import numpy as np

from .field_utils import normalize_map


CHANNEL_NAMES = [
    "obstacle",
    "goal_reward",
    "target_probability",
    "desired_occupancy",
    "risk",
    "visited",
    "agent_density",
    "communication_quality",
    "formation_template",
]

CHANNEL_INDEX = {name: idx for idx, name in enumerate(CHANNEL_NAMES)}


def empty_field(grid_size: int) -> np.ndarray:
    return np.zeros((len(CHANNEL_NAMES), grid_size, grid_size), dtype=np.float32)


def build_task_field(channel_maps: Dict[str, np.ndarray], grid_size: int) -> np.ndarray:
    field = empty_field(grid_size)
    for name, values in channel_maps.items():
        if name not in CHANNEL_INDEX:
            continue
        field[CHANNEL_INDEX[name]] = np.asarray(values, dtype=np.float32)
    return field


def adapt_task_field(field: np.ndarray, mode: str, weights: list[float] | None = None) -> np.ndarray:
    task_field = np.asarray(field, dtype=np.float32)
    if mode == "multi_channel":
        return task_field
    if mode == "task_id_only":
        warnings.warn(
            "'task_id_only' is a deprecated alias; use 'no_spatial_field' instead.",
            FutureWarning,
            stacklevel=2,
        )
        return np.zeros_like(task_field, dtype=np.float32)
    if mode == "no_spatial_field":
        return np.zeros_like(task_field, dtype=np.float32)
    if mode == "single_channel":
        if weights is None:
            weights = [1.0] * task_field.shape[0]
        weight_arr = np.asarray(weights, dtype=np.float32).reshape(-1, 1, 1)
        merged = (task_field * weight_arr).sum(axis=0, keepdims=True)
        return normalize_map(merged).astype(np.float32)
    raise ValueError(f"Unsupported observation mode: {mode}")
