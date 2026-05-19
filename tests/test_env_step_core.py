from __future__ import annotations

import numpy as np

from envs import CentralizedMultiUAVEnv
from envs.dynamics import waypoint_controller
from fields.field_utils import world_to_grid
from scripts._common import load_env_config


def _env_config(**override) -> dict:
    defaults = {"task_name": "goal_nav", "num_agents": 2}
    defaults.update(override)
    return load_env_config("configs/env/multitask.yaml", override=defaults)


def test_env_step_core_shapes_and_action_clipping():
    env = CentralizedMultiUAVEnv(_env_config())
    obs, _ = env.reset(seed=11)
    assert set(obs.keys()) == {"task_field", "agents", "task_id", "global_info"}
    assert env.action_space.shape == (env.num_agents, 2)

    next_obs, reward, terminated, truncated, info = env.step(np.full(env.action_space.shape, 10.0, dtype=np.float32))
    assert next_obs["agents"].shape == obs["agents"].shape
    assert isinstance(reward, float)
    assert terminated in {True, False}
    assert truncated in {True, False}
    assert np.all(next_obs["agents"][:, :2] >= 0.0)
    assert np.all(next_obs["agents"][:, :2] <= env.runtime_params["map_size"] + 1e-6)
    assert info["collision_count"] >= 0


def test_obstacle_collision_blocks_motion_into_obstacle_cell():
    env = CentralizedMultiUAVEnv(_env_config())
    env.reset(seed=13)
    env.state["positions"] = np.array([[0.10, 0.10], [0.80, 0.80]], dtype=np.float32)
    env.state["velocities"] = np.zeros((env.num_agents, 2), dtype=np.float32)
    env.state["obstacle_map"].fill(0.0)

    action = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    waypoints = np.clip(env.state["positions"] + action * env.runtime_params["max_waypoint_step"], 0.0, env.runtime_params["map_size"])
    proposed_positions, _ = waypoint_controller(
        env.state["positions"],
        waypoints,
        kp=float(env.config["kp"]),
        max_speed=env.runtime_params["max_speed"],
        dt=float(env.config["dt"]),
        map_size=env.runtime_params["map_size"],
    )
    obstacle_idx = world_to_grid(proposed_positions[[0]], env.grid_size, map_size=env.runtime_params["map_size"])[0]
    env.state["obstacle_map"][obstacle_idx[1], obstacle_idx[0]] = 1.0

    previous_position = env.state["positions"][0].copy()
    next_obs, _, _, _, info = env.step(action)
    assert np.allclose(next_obs["agents"][0, :2], previous_position)
    assert info["task_specific_metrics"]["goal_coverage_ratio"] >= 0.0


def test_pairwise_collision_is_counted_and_agents_do_not_advance_through_each_other():
    env = CentralizedMultiUAVEnv(_env_config())
    env.reset(seed=17)
    collision_radius = env.runtime_params["collision_radius"]
    env.state["positions"] = np.array(
        [[0.25, 0.25], [0.25 + 0.5 * collision_radius, 0.25]],
        dtype=np.float32,
    )
    env.state["velocities"] = np.zeros((env.num_agents, 2), dtype=np.float32)
    previous_positions = env.state["positions"].copy()

    next_obs, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert info["collision_count"] >= 1
    assert np.allclose(next_obs["agents"][:, :2], previous_positions)
