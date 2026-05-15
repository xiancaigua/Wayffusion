from __future__ import annotations

from copy import deepcopy

import numpy as np

from envs import CentralizedMultiUAVEnv


def stack_observations(observations: list[dict]) -> dict[str, np.ndarray]:
    keys = observations[0].keys()
    return {key: np.stack([obs[key] for obs in observations], axis=0).astype(np.float32) for key in keys}


class SyncEnvBatch:
    def __init__(self, envs: list[CentralizedMultiUAVEnv]):
        self.envs = envs
        self.num_envs = len(envs)

    def reset(self, seeds: list[int] | None = None, options: dict | None = None):
        observations = []
        infos = []
        for idx, env in enumerate(self.envs):
            seed = None if seeds is None else seeds[idx]
            obs, info = env.reset(seed=seed, options=options)
            observations.append(obs)
            infos.append(info)
        return stack_observations(observations), infos

    def step(self, actions: np.ndarray):
        next_observations = []
        rewards = []
        dones = []
        truncs = []
        infos = []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            if done:
                info = dict(info)
                info["terminal_info"] = dict(info)
                reset_obs, reset_info = env.reset()
                info["reset_info"] = reset_info
                obs = reset_obs
            next_observations.append(obs)
            rewards.append(float(reward))
            dones.append(float(done))
            truncs.append(bool(truncated))
            infos.append(info)
        return (
            stack_observations(next_observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            np.asarray(truncs, dtype=bool),
            infos,
        )


def make_env_batch(config: dict, num_envs: int, task_name: str | None = None) -> SyncEnvBatch:
    envs = []
    for env_idx in range(num_envs):
        env_config = deepcopy(config)
        env_config["seed"] = int(config.get("seed", 0)) + env_idx
        if task_name is not None:
            env_config["task_name"] = task_name
        envs.append(CentralizedMultiUAVEnv(env_config))
    return SyncEnvBatch(envs)
