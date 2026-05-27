from __future__ import annotations

import numpy as np
import torch

from policies.attention_policy import CNNAttentionPolicy
from policies.action_distribution import SquashedNormal
from policies.cnn_deepsets_policy import CNNDeepSetsPolicy
from policies.mlp_policy import MLPPolicy


def observation_to_tensor(observation: dict, device: torch.device, already_batched: bool = False) -> dict[str, torch.Tensor]:
    tensors = {}
    for key, value in observation.items():
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        if not already_batched and tensor.ndim == len(np.asarray(value).shape):
            tensor = tensor.unsqueeze(0)
        tensors[key] = tensor
    return tensors


def build_policy(policy_config: dict, observation_space, action_space):
    policy_class = policy_config.get("policy_class", "mlp")
    if policy_class == "mlp":
        return MLPPolicy(observation_space, action_space, hidden_dims=policy_config.get("hidden_dims"))
    if policy_class == "cnn_deepsets":
        return CNNDeepSetsPolicy(
            observation_space,
            action_space,
            cnn_channels=policy_config.get("cnn_channels"),
            agent_hidden_dim=int(policy_config.get("agent_hidden_dim", 64)),
            joint_hidden_dim=int(policy_config.get("joint_hidden_dim", 256)),
            decoder_hidden_dim=int(policy_config.get("decoder_hidden_dim", 128)),
            use_spatial_attention=bool(policy_config.get("use_spatial_attention", False)),
            spatial_pool_size=int(policy_config.get("spatial_pool_size", 8)),
            attention_heads=int(policy_config.get("attention_heads", 4)),
            log_std_min=float(policy_config.get("log_std_min", -1.5)),
            log_std_max=float(policy_config.get("log_std_max", 0.5)),
            log_std_init=float(policy_config.get("log_std_init", 0.0)),
        )
    if policy_class == "attention":
        return CNNAttentionPolicy(
            observation_space,
            action_space,
            embed_dim=int(policy_config.get("embed_dim", 128)),
            num_heads=int(policy_config.get("num_heads", 4)),
            num_layers=int(policy_config.get("num_layers", 2)),
            cnn_channels=policy_config.get("cnn_channels"),
        )
    raise ValueError(f"Unsupported policy class: {policy_class}")


__all__ = [
    "MLPPolicy",
    "CNNDeepSetsPolicy",
    "CNNAttentionPolicy",
    "SquashedNormal",
    "build_policy",
    "observation_to_tensor",
]
