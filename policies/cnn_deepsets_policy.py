from __future__ import annotations

import torch
from torch import nn

from policies.action_distribution import SquashedNormal


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None, dim: int) -> torch.Tensor:
    if mask is None:
        return values.mean(dim=dim)
    weights = mask.float().unsqueeze(-1)
    return (values * weights).sum(dim=dim) / torch.clamp(weights.sum(dim=dim), min=1.0)


class CNNDeepSetsPolicy(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        cnn_channels: list[int] | None = None,
        agent_hidden_dim: int = 64,
        joint_hidden_dim: int = 256,
        decoder_hidden_dim: int = 128,
    ):
        super().__init__()
        cnn_channels = cnn_channels or [16, 32, 64]
        in_channels = observation_space["task_field"].shape[0]
        conv_layers = []
        last_channels = in_channels
        for out_channels in cnn_channels:
            conv_layers += [
                nn.Conv2d(last_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            ]
            last_channels = out_channels
        conv_layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        self.field_encoder = nn.Sequential(*conv_layers)

        agent_input_dim = observation_space["agents"].shape[-1]
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_input_dim, agent_hidden_dim),
            nn.ReLU(),
            nn.Linear(agent_hidden_dim, agent_hidden_dim),
            nn.ReLU(),
        )
        task_dim = int(observation_space["task_id"].shape[0])
        global_dim = int(observation_space["global_info"].shape[0])
        self.state_mlp = nn.Sequential(
            nn.Linear(last_channels + agent_hidden_dim + task_dim + global_dim, joint_hidden_dim),
            nn.ReLU(),
            nn.Linear(joint_hidden_dim, joint_hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(agent_hidden_dim + joint_hidden_dim + task_dim + global_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, 2),
        )
        self.value_head = nn.Linear(joint_hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(2))

    def _agent_mask(self, obs: dict[str, torch.Tensor]) -> torch.Tensor | None:
        if "agent_mask" in obs:
            return obs["agent_mask"].bool()
        return None

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        field_feat = self.field_encoder(obs["task_field"])
        agent_tokens = self.agent_encoder(obs["agents"])
        agent_mask = self._agent_mask(obs)
        pooled_agents = masked_mean(agent_tokens, agent_mask, dim=1)
        task_feat = obs["task_id"].reshape(obs["task_id"].shape[0], -1)
        global_feat = obs["global_info"].reshape(obs["global_info"].shape[0], -1)
        state_feat = self.state_mlp(torch.cat([field_feat, pooled_agents, task_feat, global_feat], dim=-1))

        batch_size, num_agents, _ = agent_tokens.shape
        expanded_state = state_feat.unsqueeze(1).expand(batch_size, num_agents, -1)
        expanded_task = task_feat.unsqueeze(1).expand(batch_size, num_agents, -1)
        expanded_global = global_feat.unsqueeze(1).expand(batch_size, num_agents, -1)
        decoder_input = torch.cat([agent_tokens, expanded_state, expanded_task, expanded_global], dim=-1)
        mean = self.decoder(decoder_input)
        if agent_mask is not None:
            mean = mean * agent_mask.unsqueeze(-1).float()
        value = self.value_head(state_feat).squeeze(-1)
        return mean, value

    def get_action_and_value(self, obs: dict[str, torch.Tensor], action: torch.Tensor | None = None):
        mean, value = self(obs)
        std = self.log_std.exp().view(1, 1, 2).expand_as(mean)
        dist = SquashedNormal(mean, std)
        if action is None:
            bounded_action, raw_action = dist.rsample(return_raw=True)
        else:
            if action.ndim == 2:
                action = action.view(mean.shape[0], mean.shape[1], 2)
            bounded_action = torch.clamp(action, -1.0, 1.0)
            raw_action = dist.atanh(bounded_action)
        log_prob = dist.log_prob(bounded_action, raw_action=raw_action, reduce=False).sum(dim=-1)
        entropy = dist.entropy_estimate(bounded_action, raw_action=raw_action, reduce=False).sum(dim=-1)
        agent_mask = self._agent_mask(obs)
        if agent_mask is not None:
            log_prob = (log_prob * agent_mask.float()).sum(dim=-1)
            entropy = (entropy * agent_mask.float()).sum(dim=-1)
            bounded_action = bounded_action * agent_mask.unsqueeze(-1).float()
        else:
            log_prob = log_prob.sum(dim=-1)
            entropy = entropy.sum(dim=-1)
        return bounded_action, log_prob, entropy, value

    def act_deterministic(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        mean, _ = self(obs)
        action = SquashedNormal(mean, self.log_std.exp().view(1, 1, 2).expand_as(mean)).deterministic()
        agent_mask = self._agent_mask(obs)
        if agent_mask is not None:
            action = action * agent_mask.unsqueeze(-1).float()
        return action
