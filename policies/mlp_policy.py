from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class MLPPolicy(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dims: list[int] | None = None):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]
        field_dim = int(np.prod(observation_space["task_field"].shape))
        agent_dim = int(np.prod(observation_space["agents"].shape))
        task_dim = int(np.prod(observation_space["task_id"].shape))
        global_dim = int(np.prod(observation_space["global_info"].shape))
        input_dim = field_dim + agent_dim + task_dim + global_dim
        action_dim = int(np.prod(action_space.shape))
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(last_dim, hidden_dim), nn.ReLU()]
            last_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.actor = nn.Linear(last_dim, action_dim)
        self.critic = nn.Linear(last_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.action_shape = action_space.shape

    def _flatten_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for key in ("task_field", "agents", "task_id", "global_info"):
            parts.append(obs[key].reshape(obs[key].shape[0], -1))
        return torch.cat(parts, dim=-1)

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._flatten_obs(obs)
        features = self.backbone(x)
        mean = torch.tanh(self.actor(features)).view(features.shape[0], *self.action_shape)
        value = self.critic(features).squeeze(-1)
        return mean, value

    def get_action_and_value(self, obs: dict[str, torch.Tensor], action: torch.Tensor | None = None):
        mean, value = self(obs)
        std = self.log_std.exp().view(1, *self.action_shape).expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        if action.ndim == 2:
            action = action.view(mean.shape[0], *self.action_shape)
        clipped_action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(clipped_action).sum(dim=(-1, -2))
        entropy = dist.entropy().sum(dim=(-1, -2))
        return clipped_action, log_prob, entropy, value

    def act_deterministic(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        mean, _ = self(obs)
        return torch.clamp(mean, -1.0, 1.0)
