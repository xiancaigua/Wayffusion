from __future__ import annotations

import numpy as np

from scripts._common import load_env_config
from utils import make_task_balanced_env_batch


def _config(**override) -> dict:
    defaults = {"num_agents": 4, "max_steps": 8}
    defaults.update(override)
    return load_env_config("configs/env/multitask.yaml", override=defaults)


def _zero_actions(batch) -> np.ndarray:
    return np.zeros((batch.num_envs, *batch.envs[0].action_space.shape), dtype=np.float32)


def test_task_balanced_env_count():
    task_names = ["goal_nav", "coverage", "formation", "risk_nav"]
    batch = make_task_balanced_env_batch(_config(), task_names, envs_per_task=2)
    assert batch.num_envs == 8


def test_task_balanced_forced_task_names():
    task_names = ["goal_nav", "coverage", "formation", "risk_nav"]
    envs_per_task = 2
    batch = make_task_balanced_env_batch(_config(), task_names, envs_per_task=envs_per_task)
    forced_names = [env.forced_task_name for env in batch.envs]
    for task_name in task_names:
        assert forced_names.count(task_name) == envs_per_task


def test_task_balanced_thread_backend_reset_step():
    task_names = ["goal_nav", "coverage", "formation", "risk_nav"]
    batch = make_task_balanced_env_batch(
        _config(),
        task_names,
        envs_per_task=1,
        backend="thread",
        max_workers=4,
    )

    try:
        obs, infos = batch.reset()
        assert batch.num_envs == 4
        assert obs["task_field"].shape[0] == 4
        assert len(infos) == 4
        next_obs, rewards, dones, truncs, infos = batch.step(_zero_actions(batch))
        assert next_obs["agents"].shape[0] == 4
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert truncs.shape == (4,)
        assert len(infos) == 4
    finally:
        batch.close()
