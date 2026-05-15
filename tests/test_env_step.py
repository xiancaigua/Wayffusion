import numpy as np

from scripts._common import load_env_config
from envs import CentralizedMultiUAVEnv


def test_env_step_progress_and_metrics():
    config = load_env_config("configs/env/multitask.yaml", override={"task_name": "risk_nav"})
    env = CentralizedMultiUAVEnv(config)
    obs, _ = env.reset(seed=5)
    action = np.ones(env.action_space.shape, dtype=np.float32) * 0.25
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs["agents"].shape == obs["agents"].shape
    assert isinstance(reward, float)
    assert terminated in {True, False}
    assert truncated in {True, False}
    assert info["path_length"] >= 0.0
    assert info["risk_exposure"] >= 0.0
    assert info["collision_count"] >= 0
    assert np.all(next_obs["agents"][:, :2] >= 0.0)
    assert np.all(next_obs["agents"][:, :2] <= 1.0)
