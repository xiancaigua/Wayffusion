from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from .task_field import CHANNEL_NAMES


def save_task_field_plot(task_field: np.ndarray, output_path: str | Path, title: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    channels = task_field.shape[0]
    cols = 3
    rows = int(np.ceil(channels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows))
    axes = np.atleast_2d(axes)
    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx >= channels:
            continue
        im = ax.imshow(task_field[idx], origin="lower", cmap="viridis")
        ax.set_title(CHANNEL_NAMES[idx] if idx < len(CHANNEL_NAMES) else f"channel_{idx}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_rollout_plot(
    obstacle_map: np.ndarray,
    risk_map: np.ndarray,
    trajectories: Iterable[np.ndarray],
    waypoints: Iterable[np.ndarray],
    goals: np.ndarray | None,
    output_path: str | Path,
    title: str,
    map_size: float = 1.0,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    overlay = np.zeros((*obstacle_map.shape, 4), dtype=np.float32)
    overlay[..., 0] = risk_map
    overlay[..., 3] = np.clip(risk_map * 0.35, 0.0, 0.35)
    ax.imshow(overlay, origin="lower", extent=(0, map_size, 0, map_size))
    ax.imshow(obstacle_map, origin="lower", cmap="gray_r", alpha=0.4, extent=(0, map_size, 0, map_size))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for idx, traj in enumerate(trajectories):
        traj = np.asarray(traj)
        if len(traj) == 0:
            continue
        color = colors[idx % len(colors)]
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2)
        ax.scatter(traj[0, 0], traj[0, 1], color=color, marker="o", s=40)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, marker="x", s=60)
    for idx, waypoint in enumerate(waypoints):
        waypoint = np.asarray(waypoint)
        color = colors[idx % len(colors)]
        ax.scatter(waypoint[0], waypoint[1], color=color, marker="+", s=80)
    if goals is not None and len(goals) > 0:
        goals = np.asarray(goals)
        ax.scatter(goals[:, 0], goals[:, 1], color="gold", edgecolors="black", marker="*", s=120)
    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
