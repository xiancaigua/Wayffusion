from __future__ import annotations

import numpy as np

from envs import CentralizedMultiUAVEnv
from fields.field_utils import world_to_grid
from scripts._common import load_env_config


def test_fixed_map_keeps_base_map_size():
    config = load_env_config("configs/env/multitask.yaml", override={"task_name": "goal_nav", "scaling_mode": "fixed_map", "num_agents": 20})
    env = CentralizedMultiUAVEnv(config)
    assert np.isclose(env.runtime_params["map_size"], float(config["map_size"]))


def test_density_preserving_scales_runtime_params_with_sqrt_agent_ratio():
    config = load_env_config(
        "configs/env/multitask.yaml",
        override={"task_name": "goal_nav", "scaling_mode": "density_preserving", "num_agents": 20},
    )
    env = CentralizedMultiUAVEnv(config)
    expected_scale = np.sqrt(20 / int(config["reference_num_agents"]))
    assert np.isclose(env.runtime_params["spatial_scale"], expected_scale)
    assert np.isclose(env.runtime_params["map_size"], float(config["map_size"]) * expected_scale)
    assert np.isclose(env.runtime_params["max_speed"], float(config["max_speed"]) * expected_scale)
    assert np.isclose(env.runtime_params["max_waypoint_step"], float(config["max_waypoint_step"]) * expected_scale)
    assert np.isclose(env.runtime_params["collision_radius"], float(config["collision_radius"]) * expected_scale)


def test_density_preserving_goal_sampling_and_world_to_grid_stay_in_bounds():
    config = load_env_config(
        "configs/env/multitask.yaml",
        override={"task_name": "goal_nav", "scaling_mode": "density_preserving", "num_agents": 20},
    )
    env = CentralizedMultiUAVEnv(config)
    env.reset(seed=19)
    goals = env.current_task_state["goals"]
    map_size = env.runtime_params["map_size"]

    assert np.all(goals >= 0.0)
    assert np.all(goals <= map_size + 1e-6)

    indices = world_to_grid(goals, env.grid_size, map_size=map_size)
    assert np.all(indices >= 0)
    assert np.all(indices < env.grid_size)

    boundary = world_to_grid(np.array([[map_size, map_size]], dtype=np.float32), env.grid_size, map_size=map_size)
    assert np.array_equal(boundary[0], np.array([env.grid_size - 1, env.grid_size - 1]))
