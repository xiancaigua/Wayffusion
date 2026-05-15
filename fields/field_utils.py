from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def world_to_grid(points: np.ndarray, grid_size: int, map_size: float = 1.0) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    idx = np.clip(((pts / max(map_size, 1e-8)) * (grid_size - 1)).round().astype(np.int32), 0, grid_size - 1)
    return idx


def grid_to_world(indices: np.ndarray, grid_size: int, map_size: float = 1.0) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.float32)
    return (idx / max(grid_size - 1, 1) * map_size).astype(np.float32)


def meshgrid_xy(grid_size: int, map_size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, map_size, grid_size, dtype=np.float32)
    return np.meshgrid(axis, axis, indexing="xy")


def gaussian_map(
    centers: np.ndarray,
    grid_size: int,
    sigma: float = 0.08,
    amplitudes: np.ndarray | None = None,
    map_size: float = 1.0,
) -> np.ndarray:
    if centers is None or len(centers) == 0:
        return np.zeros((grid_size, grid_size), dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    if amplitudes is None:
        amplitudes = np.ones((centers.shape[0],), dtype=np.float32)
    amplitudes = np.asarray(amplitudes, dtype=np.float32)
    xx, yy = meshgrid_xy(grid_size, map_size=map_size)
    result = np.zeros((grid_size, grid_size), dtype=np.float32)
    for center, amp in zip(centers, amplitudes):
        dx = xx - center[0]
        dy = yy - center[1]
        result += amp * np.exp(-(dx * dx + dy * dy) / max(2.0 * sigma * sigma, 1e-6))
    return result.astype(np.float32)


def normalize_map(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    v_min = float(arr.min())
    v_max = float(arr.max())
    if v_max - v_min < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - v_min) / (v_max - v_min)).astype(np.float32)


def disk_mask(center: np.ndarray, radius: float, grid_size: int, map_size: float = 1.0) -> np.ndarray:
    xx, yy = meshgrid_xy(grid_size, map_size=map_size)
    dx = xx - float(center[0])
    dy = yy - float(center[1])
    return ((dx * dx + dy * dy) <= radius * radius).astype(np.float32)


def accumulate_disks(centers: np.ndarray, radius: float, grid_size: int, map_size: float = 1.0) -> np.ndarray:
    if centers is None or len(centers) == 0:
        return np.zeros((grid_size, grid_size), dtype=np.float32)
    result = np.zeros((grid_size, grid_size), dtype=np.float32)
    for center in np.asarray(centers, dtype=np.float32):
        result = np.maximum(result, disk_mask(center, radius, grid_size, map_size=map_size))
    return result.astype(np.float32)


def sample_points(
    rng: np.random.Generator,
    count: int,
    margin: float = 0.08,
    map_size: float = 1.0,
) -> np.ndarray:
    low = np.full((count, 2), margin * map_size, dtype=np.float32)
    high = np.full((count, 2), (1.0 - margin) * map_size, dtype=np.float32)
    return rng.uniform(low=low, high=high).astype(np.float32)


def sample_obstacle_map(
    rng: np.random.Generator,
    grid_size: int,
    density: float,
    size_range: Iterable[float],
    map_size: float = 1.0,
    area_scale: float = 1.0,
) -> np.ndarray:
    obstacle_map = np.zeros((grid_size, grid_size), dtype=np.float32)
    num_rects = max(1, int(round(density * 18 * area_scale)))
    min_size, max_size = list(size_range)
    for _ in range(num_rects):
        w = float(rng.uniform(min_size, max_size))
        h = float(rng.uniform(min_size, max_size))
        x = float(rng.uniform(0.05 * map_size, max(0.05 * map_size, 0.95 * map_size - w)))
        y = float(rng.uniform(0.05 * map_size, max(0.05 * map_size, 0.95 * map_size - h)))
        x0, y0 = world_to_grid(np.array([[x, y]], dtype=np.float32), grid_size, map_size=map_size)[0]
        x1, y1 = world_to_grid(np.array([[x + w, y + h]], dtype=np.float32), grid_size, map_size=map_size)[0]
        obstacle_map[y0 : y1 + 1, x0 : x1 + 1] = 1.0
    return obstacle_map


def sample_risk_map(
    rng: np.random.Generator,
    grid_size: int,
    blob_count: int,
    sigma: float,
    map_size: float = 1.0,
) -> np.ndarray:
    centers = sample_points(rng, blob_count, margin=0.1, map_size=map_size)
    amplitudes = rng.uniform(0.5, 1.0, size=(blob_count,)).astype(np.float32)
    return normalize_map(gaussian_map(centers, grid_size, sigma=sigma, amplitudes=amplitudes, map_size=map_size))


def softmax_map(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32) / max(temperature, 1e-6)
    arr = arr - arr.max()
    exp_arr = np.exp(arr)
    return (exp_arr / np.maximum(exp_arr.sum(), 1e-8)).astype(np.float32)
