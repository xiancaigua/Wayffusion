from __future__ import annotations

import numpy as np
import torch

from envs import CentralizedMultiUAVEnv
from policies import SquashedNormal, build_policy, observation_to_tensor
from scripts._common import load_env_config, load_generic_config


def _batched_observation(obs: dict[str, np.ndarray], batch_size: int) -> dict[str, np.ndarray]:
    return {key: np.repeat(value[None, ...], batch_size, axis=0).astype(np.float32) for key, value in obs.items()}


def _policy_paths() -> list[str]:
    return [
        "configs/policy/ppo_mlp.yaml",
        "configs/policy/ppo_cnn_deepsets.yaml",
        "configs/policy/ppo_attention.yaml",
    ]


def test_squashed_normal_log_prob_is_finite_near_action_boundaries():
    mean = torch.zeros((1, 2), dtype=torch.float32)
    std = torch.ones((1, 2), dtype=torch.float32)
    dist = SquashedNormal(mean, std)
    action = torch.tensor([[0.999999, -0.999999]], dtype=torch.float32)
    log_prob = dist.log_prob(action)
    assert log_prob.shape == (1,)
    assert torch.isfinite(log_prob).all()


def test_policy_actions_are_bounded_and_log_prob_shapes_match():
    env = CentralizedMultiUAVEnv(load_env_config("configs/env/multitask.yaml", override={"task_name": "goal_nav", "num_agents": 4}))
    observation, _ = env.reset(seed=23)
    obs_tensor = observation_to_tensor(observation, device=torch.device("cpu"))

    for policy_path in _policy_paths():
        policy = build_policy(load_generic_config(policy_path), env.observation_space, env.action_space)
        action, log_prob, entropy, value = policy.get_action_and_value(obs_tensor)
        assert action.shape == (1, env.num_agents, 2)
        assert log_prob.shape == (1,)
        assert entropy.shape == (1,)
        assert value.shape == (1,)
        assert torch.all(action <= 1.0 + 1e-6)
        assert torch.all(action >= -1.0 - 1e-6)
        assert torch.isfinite(log_prob).all()
        assert torch.isfinite(entropy).all()

        _, recomputed_log_prob, _, _ = policy.get_action_and_value(obs_tensor, action.detach())
        assert recomputed_log_prob.shape == (1,)
        assert torch.isfinite(recomputed_log_prob).all()


def test_policy_log_prob_batch_shape_matches_batch_size():
    env = CentralizedMultiUAVEnv(load_env_config("configs/env/multitask.yaml", override={"task_name": "coverage", "num_agents": 4}))
    observation, _ = env.reset(seed=29)
    obs_tensor = observation_to_tensor(_batched_observation(observation, batch_size=3), device=torch.device("cpu"), already_batched=True)

    for policy_path in _policy_paths():
        policy = build_policy(load_generic_config(policy_path), env.observation_space, env.action_space)
        action, log_prob, _, value = policy.get_action_and_value(obs_tensor)
        assert action.shape == (3, env.num_agents, 2)
        assert log_prob.shape == (3,)
        assert value.shape == (3,)

        _, eval_log_prob, _, _ = policy.get_action_and_value(obs_tensor, action.detach())
        assert eval_log_prob.shape == (3,)
        assert torch.isfinite(eval_log_prob).all()
