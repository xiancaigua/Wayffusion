from __future__ import annotations

import torch
from torch import nn

from policies.action_distribution import SquashedNormal


class CNNAttentionPolicy(nn.Module):
    """Centralized policy that lets field/task/agent tokens attend jointly.

    Compared with CNNDeepSets, this architecture keeps per-agent tokens in a
    Transformer encoder, so agents can condition on one another directly instead
    of only through a pooled summary. It is more expressive, but heavier.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        cnn_channels: list[int] | None = None,
    ):
        super().__init__()
        cnn_channels = cnn_channels or [16, 32, 64]
        in_channels = observation_space["task_field"].shape[0]
        # The field image is compressed to one token, then projected to the same
        # embedding dimension as task/global and agent tokens.
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
        self.field_proj = nn.Linear(last_channels, embed_dim)

        agent_input_dim = observation_space["agents"].shape[-1]
        task_dim = int(observation_space["task_id"].shape[0])
        global_dim = int(observation_space["global_info"].shape[0])
        # Token layout is: [field_token, task_global_token, agent_0, ..., agent_N].
        self.agent_proj = nn.Linear(agent_input_dim, embed_dim)
        self.task_proj = nn.Linear(task_dim + global_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.actor_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 2))
        self.value_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))
        self.log_std = nn.Parameter(torch.zeros(2))

    def _agent_mask(self, obs: dict[str, torch.Tensor]) -> torch.Tensor | None:
        if "agent_mask" in obs:
            return obs["agent_mask"].bool()
        return None

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode all tokens and return per-agent action means plus value."""

        batch_size, num_agents, _ = obs["agents"].shape
        field_token = self.field_proj(self.field_encoder(obs["task_field"])).unsqueeze(1)
        task_token = self.task_proj(torch.cat([obs["task_id"], obs["global_info"]], dim=-1)).unsqueeze(1)
        agent_tokens = self.agent_proj(obs["agents"])
        tokens = torch.cat([field_token, task_token, agent_tokens], dim=1)

        agent_mask = self._agent_mask(obs)
        key_padding_mask = None
        if agent_mask is not None:
            # Transformer padding masks use True for positions to ignore. The
            # first two prefix tokens are always valid.
            prefix = torch.zeros((batch_size, 2), dtype=torch.bool, device=agent_mask.device)
            key_padding_mask = torch.cat([prefix, ~agent_mask], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        encoded_agents = encoded[:, 2:, :]
        mean = torch.tanh(self.actor_head(encoded_agents))
        if agent_mask is not None:
            mean = mean * agent_mask.unsqueeze(-1).float()
        # Critic value is based on global prefix tokens, not on a single agent.
        state_token = encoded[:, :2, :].mean(dim=1)
        value = self.value_head(state_token).squeeze(-1)
        return mean, value

    def get_action_and_value(self, obs: dict[str, torch.Tensor], action: torch.Tensor | None = None):
        """Return action, joint log-prob, entropy, and critic value."""

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
        # Sum over action dimensions, then over valid agents to form the joint
        # action likelihood used by centralized PPO.
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
