import numpy as np

from scripts._common import load_env_config
from envs import CentralizedMultiUAVEnv


def test_all_tasks_share_same_task_field_shape():
    config = load_env_config("configs/env/multitask.yaml")
    for task_name in ["goal_nav", "coverage", "formation", "risk_nav"]:
        env = CentralizedMultiUAVEnv({**config, "task_name": task_name})
        obs, info = env.reset(seed=3)
        assert obs["task_field"].shape == (9, config["grid_size"], config["grid_size"])
        if task_name == "goal_nav":
            assert np.allclose(info["full_task_field"][2], 0.0)
        if task_name == "coverage":
            assert info["full_task_field"][2].sum() > 0.0
        if task_name == "formation":
            assert info["full_task_field"][8].sum() > 0.0


def test_observation_ablation_modes():
    base = load_env_config("configs/env/multitask.yaml", override={"task_name": "coverage"})
    env_single = CentralizedMultiUAVEnv({**base, "observation_mode": "single_channel"})
    obs_single, _ = env_single.reset(seed=9)
    assert obs_single["task_field"].shape[0] == 1

    env_id = CentralizedMultiUAVEnv({**base, "observation_mode": "task_id_only"})
    obs_id, _ = env_id.reset(seed=9)
    assert obs_id["task_field"].shape[0] == 9
    assert np.allclose(obs_id["task_field"], 0.0)
