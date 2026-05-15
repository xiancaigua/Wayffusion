from __future__ import annotations

from typing import Iterable

import numpy as np


def compute_intrinsic_score(task_name: str, metrics: dict, spatial_scale: float = 1.0) -> float:
    if task_name in {"goal_nav", "risk_nav"}:
        coverage = float(metrics.get("goal_coverage_ratio", metrics.get("task_success_rate", 0.0)))
        collisions = float(metrics.get("collision_rate", 0.0))
        norm_scale = max(spatial_scale, 1.0)
        path_penalty = min(float(metrics.get("path_length", 0.0)), 5.0 * norm_scale) / max(5.0 * norm_scale, 1e-6)
        risk_penalty = min(float(metrics.get("cumulative_risk_exposure", 0.0)), 5.0 * norm_scale) / max(5.0 * norm_scale, 1e-6)
        return float(np.clip(0.65 * coverage + 0.2 * (1.0 - collisions) + 0.1 * (1.0 - path_penalty) + 0.05 * (1.0 - risk_penalty), 0.0, 1.0))
    if task_name == "coverage":
        return float(
            np.clip(
                0.5 * float(metrics.get("coverage_ratio", 0.0))
                + 0.3 * float(metrics.get("accumulated_detection_probability", 0.0))
                + 0.2 * (1.0 - float(metrics.get("repeated_coverage_ratio", 1.0))),
                0.0,
                1.0,
            )
        )
    if task_name == "formation":
        formation_error = float(metrics.get("formation_error", 1.0))
        angular = float(metrics.get("angular_coverage_uniformity", 0.0))
        radius_error = float(metrics.get("radius_error", 1.0))
        scale = max(0.25 * spatial_scale, 1e-6)
        return float(np.clip(0.45 * (1.0 - min(formation_error / scale, 1.0)) + 0.35 * angular + 0.2 * (1.0 - min(radius_error / scale, 1.0)), 0.0, 1.0))
    return 0.0


def compute_reference_normalization(
    return_value: float,
    random_return: float,
    heuristic_return: float,
    eps: float = 1e-6,
    clip_value: float = 5.0,
) -> dict[str, float | str]:
    random_return = float(random_return)
    heuristic_return = float(heuristic_return)
    return_value = float(return_value)
    if heuristic_return >= random_return:
        lower_return = random_return
        upper_return = heuristic_return
        best_method = "heuristic"
        worst_method = "random"
    else:
        lower_return = heuristic_return
        upper_return = random_return
        best_method = "random"
        worst_method = "heuristic"
    reference_gap = float(upper_return - lower_return)
    raw_score = float((return_value - lower_return) / max(reference_gap, eps))
    clipped_score = float(np.clip(raw_score, -clip_value, clip_value))
    return {
        "normalized_score": clipped_score,
        "normalized_score_raw": raw_score,
        "reference_random_return": random_return,
        "reference_heuristic_return": heuristic_return,
        "reference_lower_return": lower_return,
        "reference_upper_return": upper_return,
        "reference_gap": reference_gap,
        "reference_best_method": best_method,
        "reference_worst_method": worst_method,
        "reference_order_flipped": float(best_method != "heuristic"),
    }


def compute_reference_normalized_score(
    return_value: float,
    random_return: float,
    heuristic_return: float,
    eps: float = 1e-6,
    clip_value: float = 5.0,
) -> float:
    return float(
        compute_reference_normalization(
            return_value=return_value,
            random_return=random_return,
            heuristic_return=heuristic_return,
            eps=eps,
            clip_value=clip_value,
        )["normalized_score"]
    )


def summarize_episode_metrics(records: Iterable[dict]) -> dict:
    records = list(records)
    if not records:
        return {}
    keys = sorted({key for record in records for key in record.keys()})
    summary = {}
    for key in keys:
        values = [record[key] for record in records if key in record]
        if not values:
            continue
        if isinstance(values[0], (int, float, np.floating)):
            summary[key] = float(np.mean(values))
        else:
            summary[key] = values[-1]
    return summary
