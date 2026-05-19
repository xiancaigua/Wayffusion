from __future__ import annotations

import numpy as np


def clip_norm(vectors: np.ndarray, max_norm: float) -> np.ndarray:
    vec = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vec, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.maximum(norms, 1e-8))
    return (vec * scale).astype(np.float32)


def waypoint_controller(
    positions: np.ndarray,
    waypoints: np.ndarray,
    kp: float,
    max_speed: float,
    dt: float,
    map_size: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    error = np.asarray(waypoints, dtype=np.float32) - np.asarray(positions, dtype=np.float32)
    velocities = clip_norm(kp * error, max_speed)
    next_positions = np.clip(np.asarray(positions, dtype=np.float32) + dt * velocities, 0.0, float(map_size))
    return next_positions.astype(np.float32), velocities.astype(np.float32)
