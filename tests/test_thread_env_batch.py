from __future__ import annotations

import numpy as np

from scripts._common import load_env_config
from utils import make_env_batch


def _config(**override) -> dict:
    defaults = {
        "task_name": "goal_nav",
        "task_names": ["goal_nav"],
        "task_sampling_probs": {"goal_nav": 1.0, "coverage": 0.0, "formation": 0.0, "risk_nav": 0.0},
        "num_agents": 4,
        "max_steps": 8,
    }
    defaults.update(override)
    return load_env_config("configs/env/multitask.yaml", override=defaults)


def _zero_actions(batch) -> np.ndarray:
    return np.zeros((batch.num_envs, *batch.envs[0].action_space.shape), dtype=np.float32)


def test_thread_env_batch_reset_step_basic_shapes():
    num_envs = 2
    config = _config()
    sync_batch = make_env_batch(config, num_envs, backend="sync")
    thread_batch = make_env_batch(config, num_envs, backend="thread", max_workers=2)

    try:
        sync_obs, _ = sync_batch.reset()
        thread_obs, infos = thread_batch.reset()
        assert set(thread_obs.keys()) == set(sync_obs.keys())
        assert len(infos) == num_envs
        assert thread_obs["task_field"].shape[0] == num_envs
        assert thread_obs["agents"].shape[0] == num_envs
        assert thread_obs["task_id"].shape[0] == num_envs
        assert thread_obs["global_info"].shape[0] == num_envs

        next_obs, rewards, dones, truncs, infos = thread_batch.step(_zero_actions(thread_batch))
        assert next_obs["task_field"].shape[0] == num_envs
        assert next_obs["agents"].shape[0] == num_envs
        assert rewards.shape == (num_envs,)
        assert dones.shape == (num_envs,)
        assert truncs.shape == (num_envs,)
        assert len(infos) == num_envs
    finally:
        thread_batch.close()


def test_thread_env_batch_preserves_env_order():
    num_envs = 3
    batch = make_env_batch(_config(), num_envs, backend="thread", max_workers=3)

    try:
        obs, infos = batch.reset(seeds=[101, 102, 103])
        assert len(infos) == num_envs
        for idx, env in enumerate(batch.envs):
            assert np.allclose(obs["agents"][idx], env.last_observation["agents"])
            assert infos[idx]["task_name"] == env.current_task.name

        next_obs, _, _, _, infos = batch.step(_zero_actions(batch))
        assert len(infos) == num_envs
        for idx, env in enumerate(batch.envs):
            assert np.allclose(next_obs["agents"][idx], env.last_observation["agents"])
            assert infos[idx]["task_name"] == env.current_task.name
    finally:
        batch.close()


def test_thread_env_batch_auto_resets_done_envs():
    batch = make_env_batch(_config(max_steps=1), 2, backend="thread", max_workers=2)

    try:
        batch.reset()
        next_obs, rewards, dones, truncs, infos = batch.step(_zero_actions(batch))
        assert rewards.shape == (2,)
        assert np.all(dones == 1.0)
        assert np.all(truncs)
        for info in infos:
            assert "terminal_info" in info
            assert "reset_info" in info
        assert next_obs["task_field"].shape[0] == 2
        assert next_obs["agents"].shape[0] == 2
    finally:
        batch.close()
