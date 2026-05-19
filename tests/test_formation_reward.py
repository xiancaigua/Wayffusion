from __future__ import annotations

from copy import deepcopy

import numpy as np

from scripts._common import load_env_config
from tasks.formation import FormationTask


def test_radius_penalty_gets_more_negative_when_radius_error_worsens():
    config = load_env_config("configs/env/multitask.yaml")
    task = FormationTask(config)
    center = np.array([0.5, 0.5], dtype=np.float32)
    radius = float(config["formation_radius"])
    slots = task._template_slots("circle", center, 4, radius, map_size=1.0)
    base_task_state = {
        "template": "circle",
        "target_position": center,
        "target_velocity": np.zeros(2, dtype=np.float32),
        "radius": radius,
        "slots": slots,
        "last_error": 1.0,
    }
    common_env_state = {"formation_error_history": [], "spatial_scale": 1.0, "step_count": 5}
    transition_info = {
        "pair_collision_count": 0,
        "obstacle_collision_count": 0,
        "path_length_delta": 0.0,
        "step_risk_exposure": 0.0,
        "step_safety_violations": 0,
        "num_agents": 4,
        "spatial_scale": 1.0,
        "max_step_distance": 0.1,
    }

    near_env_state = {**common_env_state, "positions": slots.copy()}
    far_env_state = {**common_env_state, "positions": center[None, :] + (slots - center[None, :]) * 1.8}

    near_result = task.compute_reward(deepcopy(base_task_state), {}, near_env_state, transition_info)
    far_result = task.compute_reward(deepcopy(base_task_state), {}, far_env_state, transition_info)
    far_metrics = task.get_metrics(deepcopy(base_task_state), far_env_state)

    assert near_result.components["radius_penalty"] <= 0.0
    assert far_result.components["radius_penalty"] < near_result.components["radius_penalty"]
    assert far_result.reward < near_result.reward
    assert far_metrics["radius_error"] >= 0.0
