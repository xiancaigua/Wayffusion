from __future__ import annotations

from copy import deepcopy

import numpy as np

from envs import CentralizedMultiUAVEnv
from envs.rewards import common_reward
from scripts._common import load_env_config


def test_common_penalties_are_non_positive():
    config = load_env_config("configs/env/multitask.yaml")
    reward, components = common_reward(
        config,
        {
            "pair_collision_count": 2,
            "obstacle_collision_count": 1,
            "path_length_delta": 0.4,
            "step_safety_violations": 1,
            "step_risk_exposure": 0.6,
            "num_agents": 4,
            "spatial_scale": 1.0,
            "max_step_distance": 0.1,
        },
    )
    assert reward <= 0.0
    assert components["collision_penalty"] <= 0.0
    assert components["path_penalty"] <= 0.0


def test_goal_nav_progress_reward_is_positive_when_agents_move_closer():
    env = CentralizedMultiUAVEnv(load_env_config("configs/env/multitask.yaml", override={"task_name": "goal_nav", "num_agents": 2}))
    env.reset(seed=5)
    task = env.current_task
    task_state = deepcopy(env.current_task_state)
    prev_state = env._snapshot_state()
    env_state = env._snapshot_state()
    goal = task_state["goals"][0]
    direction = goal - env_state["positions"][0]
    direction = direction / max(np.linalg.norm(direction), 1e-6)
    env_state["positions"][0] = env_state["positions"][0] + 0.02 * direction

    result = task.compute_reward(
        task_state,
        prev_state,
        env_state,
        {
            "pair_collision_count": 0,
            "obstacle_collision_count": 0,
            "path_length_delta": 0.0,
            "step_risk_exposure": 0.0,
            "step_safety_violations": 0,
            "num_agents": env.num_agents,
            "spatial_scale": env.runtime_params["spatial_scale"],
            "max_step_distance": env.runtime_params["max_speed"] * float(env.config["dt"]),
        },
    )
    assert result.components["task_progress_reward"] > 0.0


def test_goal_nav_progress_ignores_goals_already_reached():
    env = CentralizedMultiUAVEnv(load_env_config("configs/env/multitask.yaml", override={"task_name": "goal_nav", "num_agents": 2}))
    env.reset(seed=5)
    task = env.current_task
    prev_state = env._snapshot_state()
    env_state = env._snapshot_state()
    goals = np.asarray([[0.2, 0.2], [0.8, 0.8]], dtype=np.float32)
    previous_positions = np.asarray([[0.2, 0.2], [0.72, 0.8]], dtype=np.float32)
    env_state["positions"] = np.asarray([[0.3, 0.3], [0.73, 0.8]], dtype=np.float32)
    task_state = {
        **deepcopy(env.current_task_state),
        "goals": goals,
        "goal_reached": np.asarray([True, False], dtype=bool),
        "goal_progress": task._goal_cost(goals[1:], previous_positions),
        "last_goal_coverage_ratio": 0.5,
        "success_bonus_paid": False,
    }

    result = task.compute_reward(
        task_state,
        prev_state,
        env_state,
        {
            "pair_collision_count": 0,
            "obstacle_collision_count": 0,
            "path_length_delta": 0.0,
            "step_risk_exposure": 0.0,
            "step_safety_violations": 0,
            "num_agents": env.num_agents,
            "spatial_scale": env.runtime_params["spatial_scale"],
            "max_step_distance": env.runtime_params["max_speed"] * float(env.config["dt"]),
        },
    )

    assert result.components["task_progress_reward"] > 0.0
    assert task_state["goal_progress"] == task._goal_cost(goals[1:], env_state["positions"])


def test_coverage_new_coverage_component_is_positive():
    env = CentralizedMultiUAVEnv(load_env_config("configs/env/multitask.yaml", override={"task_name": "coverage"}))
    env.reset(seed=3)
    task = env.current_task
    task_state = deepcopy(env.current_task_state)
    prev_state = env._snapshot_state()
    env_state = env._snapshot_state()
    env_state["visit_count_map"] = task_state["coverage_demand"].copy()
    env_state["step_coverage_mask"] = task_state["coverage_demand"].copy()
    result = task.compute_reward(
        task_state,
        prev_state,
        env_state,
        {
            "pair_collision_count": 0,
            "obstacle_collision_count": 0,
            "path_length_delta": 0.0,
            "step_risk_exposure": 0.0,
            "step_safety_violations": 0,
            "num_agents": env.num_agents,
            "spatial_scale": env.runtime_params["spatial_scale"],
            "max_step_distance": env.runtime_params["max_speed"] * float(env.config["dt"]),
        },
    )
    assert result.components["coverage_reward"] > 0.0


def test_coverage_demand_quantile_controls_demand_area():
    base_config = load_env_config("configs/env/multitask.yaml", override={"task_name": "coverage"})
    focused_config = load_env_config(
        "configs/env/multitask.yaml",
        override={"task_name": "coverage", "coverage": {"demand_quantile": 0.75}},
    )
    base_env = CentralizedMultiUAVEnv(base_config)
    focused_env = CentralizedMultiUAVEnv(focused_config)
    base_env.reset(seed=12)
    focused_env.reset(seed=12)

    base_area = float(base_env.current_task_state["coverage_demand"].mean())
    focused_area = float(focused_env.current_task_state["coverage_demand"].mean())
    assert focused_area < base_area


def test_coverage_failure_penalty_only_on_unsuccessful_timeout():
    env = CentralizedMultiUAVEnv(
        load_env_config(
            "configs/env/multitask.yaml",
            override={
                "task_name": "coverage",
                "reward_weights": {
                    "coverage": {
                        "new_coverage": 0.0,
                        "high_probability": 0.0,
                        "coverage_level": 0.0,
                        "coverage_shortfall": 0.0,
                        "repeated_coverage": 0.0,
                        "success_bonus": 0.0,
                        "failure_penalty": -7.0,
                    }
                },
            },
        )
    )
    env.reset(seed=3)
    task = env.current_task
    task_state = deepcopy(env.current_task_state)
    prev_state = env._snapshot_state()
    env_state = env._snapshot_state()
    env_state["step_count"] = int(env_state["max_steps"])
    env_state["step_coverage_mask"] = np.zeros_like(env_state["step_coverage_mask"], dtype=np.float32)
    result = task.compute_reward(
        task_state,
        prev_state,
        env_state,
        {
            "pair_collision_count": 0,
            "obstacle_collision_count": 0,
            "path_length_delta": 0.0,
            "step_risk_exposure": 0.0,
            "step_safety_violations": 0,
            "num_agents": env.num_agents,
            "spatial_scale": env.runtime_params["spatial_scale"],
            "max_step_distance": env.runtime_params["max_speed"] * float(env.config["dt"]),
        },
    )
    assert result.components["coverage_failure_penalty"] == -7.0

    env_state["visit_count_map"] = task_state["coverage_demand"].copy()
    env_state["step_coverage_mask"] = task_state["coverage_demand"].copy()
    task_state["last_coverage_ratio"] = 0.0
    success_result = task.compute_reward(
        task_state,
        prev_state,
        env_state,
        {
            "pair_collision_count": 0,
            "obstacle_collision_count": 0,
            "path_length_delta": 0.0,
            "step_risk_exposure": 0.0,
            "step_safety_violations": 0,
            "num_agents": env.num_agents,
            "spatial_scale": env.runtime_params["spatial_scale"],
            "max_step_distance": env.runtime_params["max_speed"] * float(env.config["dt"]),
        },
    )
    assert success_result.success
    assert success_result.components["coverage_failure_penalty"] == 0.0


def test_coverage_milestone_reward_pays_once():
    env = CentralizedMultiUAVEnv(
        load_env_config(
            "configs/env/multitask.yaml",
            override={
                "task_name": "coverage",
                "reward_weights": {
                    "coverage": {
                        "new_coverage": 0.0,
                        "high_probability": 0.0,
                        "coverage_level": 0.0,
                        "coverage_shortfall": 0.0,
                        "repeated_coverage": 0.0,
                        "success_bonus": 0.0,
                        "failure_penalty": 0.0,
                        "milestone_thresholds": [0.2, 0.4],
                        "milestone_bonuses": [2.0, 3.0],
                    }
                },
            },
        )
    )
    env.reset(seed=3)
    task = env.current_task
    task_state = deepcopy(env.current_task_state)
    prev_state = env._snapshot_state()
    env_state = env._snapshot_state()
    demand = task_state["coverage_demand"].astype(bool)
    demand_indices = np.argwhere(demand)
    first_count = max(1, int(np.ceil(0.45 * len(demand_indices))))
    env_state["visit_count_map"] = np.zeros_like(env_state["visit_count_map"], dtype=np.float32)
    env_state["step_coverage_mask"] = np.zeros_like(env_state["step_coverage_mask"], dtype=np.float32)
    for y, x in demand_indices[:first_count]:
        env_state["visit_count_map"][y, x] = 1.0
        env_state["step_coverage_mask"][y, x] = 1.0

    transition = {
        "pair_collision_count": 0,
        "obstacle_collision_count": 0,
        "path_length_delta": 0.0,
        "step_risk_exposure": 0.0,
        "step_safety_violations": 0,
        "num_agents": env.num_agents,
        "spatial_scale": env.runtime_params["spatial_scale"],
        "max_step_distance": env.runtime_params["max_speed"] * float(env.config["dt"]),
    }
    result = task.compute_reward(task_state, prev_state, env_state, transition)
    assert result.components["coverage_milestone_reward"] == 5.0

    repeated_result = task.compute_reward(task_state, prev_state, env_state, transition)
    assert repeated_result.components["coverage_milestone_reward"] == 0.0


def test_coverage_terminal_repeated_penalty_only_on_timeout():
    env = CentralizedMultiUAVEnv(
        load_env_config(
            "configs/env/multitask.yaml",
            override={
                "task_name": "coverage",
                "reward_weights": {
                    "coverage": {
                        "new_coverage": 0.0,
                        "high_probability": 0.0,
                        "coverage_level": 0.0,
                        "coverage_shortfall": 0.0,
                        "repeated_coverage": 0.0,
                        "terminal_repeated_coverage": -5.0,
                        "success_bonus": 0.0,
                        "failure_penalty": 0.0,
                    }
                },
            },
        )
    )
    env.reset(seed=3)
    task = env.current_task
    task_state = deepcopy(env.current_task_state)
    prev_state = env._snapshot_state()
    env_state = env._snapshot_state()
    env_state["visit_count_map"] = 2.0 * task_state["coverage_demand"].copy()
    env_state["step_coverage_mask"] = task_state["coverage_demand"].copy()
    transition = {
        "pair_collision_count": 0,
        "obstacle_collision_count": 0,
        "path_length_delta": 0.0,
        "step_risk_exposure": 0.0,
        "step_safety_violations": 0,
        "num_agents": env.num_agents,
        "spatial_scale": env.runtime_params["spatial_scale"],
        "max_step_distance": env.runtime_params["max_speed"] * float(env.config["dt"]),
    }

    non_terminal = task.compute_reward(task_state, prev_state, env_state, transition)
    assert non_terminal.components["terminal_repeated_coverage_penalty"] == 0.0

    env_state["step_count"] = int(env_state["max_steps"])
    terminal = task.compute_reward(task_state, prev_state, env_state, transition)
    assert terminal.components["terminal_repeated_coverage_penalty"] < 0.0


def test_coverage_demand_revisit_penalties_are_configurable():
    env = CentralizedMultiUAVEnv(
        load_env_config(
            "configs/env/multitask.yaml",
            override={
                "task_name": "coverage",
                "reward_weights": {
                    "coverage": {
                        "new_coverage": 0.0,
                        "high_probability": 0.0,
                        "coverage_level": 0.0,
                        "coverage_shortfall": 0.0,
                        "repeated_coverage": 0.0,
                        "repeated_demand_coverage": -3.0,
                        "terminal_repeated_coverage": 0.0,
                        "terminal_revisit_excess": -2.0,
                        "success_bonus": 0.0,
                        "failure_penalty": 0.0,
                    }
                },
            },
        )
    )
    env.reset(seed=14)
    task = env.current_task
    task_state = deepcopy(env.current_task_state)
    prev_state = env._snapshot_state()
    env_state = env._snapshot_state()
    demand = task_state["coverage_demand"].astype(np.float32)
    prev_state["visit_count_map"] = demand.copy()
    env_state["visit_count_map"] = 2.0 * demand.copy()
    env_state["step_coverage_mask"] = demand.copy()
    transition = {
        "pair_collision_count": 0,
        "obstacle_collision_count": 0,
        "path_length_delta": 0.0,
        "step_risk_exposure": 0.0,
        "step_safety_violations": 0,
        "num_agents": env.num_agents,
        "spatial_scale": env.runtime_params["spatial_scale"],
        "max_step_distance": env.runtime_params["max_speed"] * float(env.config["dt"]),
    }

    result = task.compute_reward(task_state, prev_state, env_state, transition)
    assert result.components["repeated_demand_coverage_penalty"] < 0.0
    assert result.components["terminal_revisit_excess_penalty"] == 0.0

    env_state["step_count"] = int(env_state["max_steps"])
    terminal = task.compute_reward(task_state, prev_state, env_state, transition)
    assert terminal.metrics["demand_revisit_excess"] > 0.0
    assert terminal.components["terminal_revisit_excess_penalty"] < 0.0


def test_risk_nav_high_exposure_produces_negative_task_penalty():
    env = CentralizedMultiUAVEnv(load_env_config("configs/env/multitask.yaml", override={"task_name": "risk_nav"}))
    env.reset(seed=7)
    task = env.current_task
    task_state = deepcopy(env.current_task_state)
    prev_state = env._snapshot_state()
    env_state = env._snapshot_state()
    result = task.compute_reward(
        task_state,
        prev_state,
        env_state,
        {
            "pair_collision_count": 0,
            "obstacle_collision_count": 0,
            "path_length_delta": 0.0,
            "step_risk_exposure": 5.0,
            "step_safety_violations": 0,
            "num_agents": env.num_agents,
            "spatial_scale": env.runtime_params["spatial_scale"],
            "max_step_distance": env.runtime_params["max_speed"] * float(env.config["dt"]),
        },
    )
    assert result.components["risk_task_penalty"] < 0.0
