from scripts._common import load_env_config
from envs import CentralizedMultiUAVEnv


def test_env_reset_shapes_and_seed():
    config = load_env_config("configs/env/multitask.yaml")
    env_a = CentralizedMultiUAVEnv(config)
    env_b = CentralizedMultiUAVEnv(config)
    obs_a, info_a = env_a.reset(seed=11, options={"task_name": "goal_nav"})
    obs_b, info_b = env_b.reset(seed=11, options={"task_name": "goal_nav"})

    assert set(obs_a.keys()) == {"task_field", "agents", "task_id", "global_info"}
    assert obs_a["task_field"].shape == (9, config["grid_size"], config["grid_size"])
    assert obs_a["agents"].shape == (config["num_agents"], 6)
    assert obs_a["task_id"].shape == (4,)
    assert obs_a["global_info"].shape == (5,)
    assert info_a["task_name"] == "goal_nav"
    assert info_b["task_name"] == "goal_nav"
    assert (obs_a["agents"] == obs_b["agents"]).all()
    assert (obs_a["task_field"] == obs_b["task_field"]).all()
