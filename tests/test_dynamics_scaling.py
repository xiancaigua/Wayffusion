from __future__ import annotations

import numpy as np

from envs import CentralizedMultiUAVEnv
from envs.dynamics import waypoint_controller
from scripts._common import load_env_config


def test_waypoint_controller_respects_map_size_argument():
    positions = np.array([[0.95, 0.95]], dtype=np.float32)
    waypoints = np.array([[2.0, 2.0]], dtype=np.float32)
    next_positions, velocities = waypoint_controller(
        positions,
        waypoints,
        kp=10.0,
        max_speed=10.0,
        dt=1.0,
        map_size=1.5,
    )

    assert np.all(next_positions <= 1.5 + 1e-6)
    assert np.any(next_positions > 1.0)
    assert np.all(np.isfinite(velocities))


def test_fixed_map_waypoint_controller_keeps_positions_inside_unit_map():
    positions = np.array([[0.95, 0.95]], dtype=np.float32)
    waypoints = np.array([[2.0, 2.0]], dtype=np.float32)
    next_positions, _ = waypoint_controller(
        positions,
        waypoints,
        kp=10.0,
        max_speed=10.0,
        dt=1.0,
    )
    assert np.all(next_positions <= 1.0 + 1e-6)


def test_density_preserving_env_allows_motion_beyond_one():
    config = load_env_config(
        "configs/env/multitask.yaml",
        override={"task_name": "goal_nav", "num_agents": 16, "scaling_mode": "density_preserving"},
    )
    env = CentralizedMultiUAVEnv(config)
    env.reset(seed=5)
    assert env.runtime_params["map_size"] > 1.0

    positions = np.array(
        [[0.995, 0.995]]
        + [[0.2 + 0.2 * (idx % 4), 0.2 + 0.2 * (idx // 4)] for idx in range(1, env.num_agents)],
        dtype=np.float32,
    )
    env.state["positions"] = positions
    env.state["velocities"] = np.zeros_like(positions, dtype=np.float32)
    env.state["obstacle_map"].fill(0.0)

    action = np.ones(env.action_space.shape, dtype=np.float32)
    next_obs, _, _, _, _ = env.step(action)
    assert np.any(next_obs["agents"][:, :2] > 1.0)
    assert np.all(next_obs["agents"][:, :2] <= env.runtime_params["map_size"] + 1e-6)
