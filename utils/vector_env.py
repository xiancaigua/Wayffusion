from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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


class ThreadEnvBatch:
    """Threaded counterpart to SyncEnvBatch with the same public interface.

    Each environment instance is still an independent CentralizedMultiUAVEnv.
    The only difference from SyncEnvBatch is that reset/step calls are submitted
    to a ThreadPoolExecutor and then collected in submission order.
    """

    def __init__(self, envs: list[CentralizedMultiUAVEnv], max_workers: int | None = None):
        self.envs = envs
        self.num_envs = len(envs)
        self.max_workers = max_workers or len(envs)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def reset(self, seeds: list[int] | None = None, options: dict | None = None):
        futures = []
        for idx, env in enumerate(self.envs):
            seed = None if seeds is None else seeds[idx]
            futures.append(self.executor.submit(env.reset, seed=seed, options=options))
        results = [future.result() for future in futures]
        observations = [obs for obs, _ in results]
        infos = [info for _, info in results]
        return stack_observations(observations), infos

    def step(self, actions: np.ndarray):
        futures = [self.executor.submit(self._step_env, env, action) for env, action in zip(self.envs, actions)]
        results = [future.result() for future in futures]
        next_observations = [result[0] for result in results]
        rewards = [result[1] for result in results]
        dones = [result[2] for result in results]
        truncs = [result[3] for result in results]
        infos = [result[4] for result in results]
        return (
            stack_observations(next_observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            np.asarray(truncs, dtype=bool),
            infos,
        )

    def _step_env(self, env: CentralizedMultiUAVEnv, action: np.ndarray):
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        if done:
            info = dict(info)
            info["terminal_info"] = dict(info)
            reset_obs, reset_info = env.reset()
            info["reset_info"] = reset_info
            obs = reset_obs
        return obs, float(reward), float(done), bool(truncated), info

    def close(self) -> None:
        self.executor.shutdown(wait=True)
        for env in self.envs:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()


def _build_batch(envs: list[CentralizedMultiUAVEnv], backend: str, max_workers: int | None = None):
    if backend == "sync":
        return SyncEnvBatch(envs)
    if backend == "thread":
        return ThreadEnvBatch(envs, max_workers=max_workers)
    raise ValueError(f"Unsupported backend: {backend}")


def make_env_batch(
    config: dict,
    num_envs: int,
    task_name: str | None = None,
    backend: str = "sync",
    max_workers: int | None = None,
):
    envs = []
    for env_idx in range(num_envs):
        env_config = deepcopy(config)
        env_config["seed"] = int(config.get("seed", 0)) + env_idx
        if task_name is not None:
            env_config["task_name"] = task_name
        envs.append(CentralizedMultiUAVEnv(env_config))
    return _build_batch(envs, backend=backend, max_workers=max_workers)


def make_task_balanced_env_batch(
    config: dict,
    task_names: list[str],
    envs_per_task: int,
    backend: str = "sync",
    max_workers: int | None = None,
):
    if not task_names:
        raise ValueError("task_names must not be empty.")
    if int(envs_per_task) < 1:
        raise ValueError("envs_per_task must be >= 1.")

    envs = []
    for task_idx, task_name in enumerate(task_names):
        for local_idx in range(int(envs_per_task)):
            env_idx = task_idx * int(envs_per_task) + local_idx
            env_config = deepcopy(config)
            env_config["seed"] = int(config.get("seed", 0)) + env_idx
            env_config["task_name"] = task_name
            envs.append(CentralizedMultiUAVEnv(env_config))
    return _build_batch(envs, backend=backend, max_workers=max_workers)
