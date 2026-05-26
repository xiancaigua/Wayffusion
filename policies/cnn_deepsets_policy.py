from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from policies.action_distribution import SquashedNormal


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None, dim: int) -> torch.Tensor:
    """Mean-pool token tensors while ignoring padded agents when a mask exists."""

    if mask is None:
        return values.mean(dim=dim)
    weights = mask.float().unsqueeze(-1)
    return (values * weights).sum(dim=dim) / torch.clamp(weights.sum(dim=dim), min=1.0)


class CNNDeepSetsPolicy(nn.Module):
    """Variable-N centralized policy using CNN field encoding and DeepSets.

    Architecture:
    - CNN encodes the spatial task field into one global field feature.
    - A shared MLP encodes each UAV independently into an agent token.
    - Masked mean pooling creates a permutation-invariant swarm summary.
    - A joint state MLP produces the critic feature.
    - A per-agent decoder combines each local token with the joint feature and
      emits that UAV's 2-D waypoint action mean.

    This is the main scalable policy because it can handle different numbers of
    agents as long as padded entries are accompanied by `agent_mask`.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        cnn_channels: list[int] | None = None,
        agent_hidden_dim: int = 64,
        joint_hidden_dim: int = 256,
        decoder_hidden_dim: int = 128,
        use_spatial_attention: bool = False,
        spatial_pool_size: int = 8,
        attention_heads: int = 4,
        log_std_min: float = -1.5,
        log_std_max: float = 0.5,
    ):
        super().__init__()
        cnn_channels = cnn_channels or [16, 32, 64]
        self.use_spatial_attention = bool(use_spatial_attention)
        self.spatial_pool_size = int(spatial_pool_size)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        in_channels = observation_space["task_field"].shape[0]
        # Strided convolutions compress the fixed-resolution task field while
        # preserving the idea that nearby cells form local spatial patterns.
        conv_layers = []
        last_channels = in_channels
        for out_channels in cnn_channels:
            conv_layers += [
                nn.Conv2d(last_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            ]
            last_channels = out_channels
        self.field_encoder = nn.Sequential(*conv_layers)
        self.field_global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())

        agent_input_dim = observation_space["agents"].shape[-1]
        # Agent encoder is shared across UAVs. Sharing makes the policy
        # permutation-compatible and avoids parameters growing with N.
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_input_dim, agent_hidden_dim),
            nn.ReLU(),
            nn.Linear(agent_hidden_dim, agent_hidden_dim),
            nn.ReLU(),
        )
        spatial_context_dim = 0
        if self.use_spatial_attention:
            self.coord_proj = nn.Linear(2, last_channels)
            self.agent_query = nn.Linear(agent_hidden_dim, last_channels)
            self.spatial_attention = nn.MultiheadAttention(
                embed_dim=last_channels,
                num_heads=int(attention_heads),
                batch_first=True,
            )
            spatial_context_dim = last_channels
        task_dim = int(observation_space["task_id"].shape[0])
        global_dim = int(observation_space["global_info"].shape[0])
        # The critic sees a global summary, not individual per-agent decisions.
        # This matches the centralized-agent formulation.
        self.state_mlp = nn.Sequential(
            nn.Linear(last_channels + agent_hidden_dim + spatial_context_dim + task_dim + global_dim, joint_hidden_dim),
            nn.ReLU(),
            nn.Linear(joint_hidden_dim, joint_hidden_dim),
            nn.ReLU(),
        )
        # The actor is decoded per agent from local token + global context. That
        # keeps actions individualized while still coordinated by state_feat.
        self.decoder = nn.Sequential(
            nn.Linear(agent_hidden_dim + spatial_context_dim + joint_hidden_dim + task_dim + global_dim, decoder_hidden_dim),
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
        """Return action means [B, N, 2] and state values [B]."""

        field_map = self.field_encoder(obs["task_field"])
        field_feat = self.field_global_pool(field_map)
        agent_tokens = self.agent_encoder(obs["agents"])
        agent_mask = self._agent_mask(obs)
        spatial_context = None
        if self.use_spatial_attention:
            pooled_map = F.adaptive_avg_pool2d(field_map, (self.spatial_pool_size, self.spatial_pool_size))
            batch_size, channels, height, width = pooled_map.shape
            spatial_tokens = pooled_map.flatten(2).transpose(1, 2)
            ys = torch.linspace(-1.0, 1.0, height, device=pooled_map.device, dtype=pooled_map.dtype)
            xs = torch.linspace(-1.0, 1.0, width, device=pooled_map.device, dtype=pooled_map.dtype)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            coords = torch.stack([grid_x, grid_y], dim=-1).reshape(1, height * width, 2)
            spatial_tokens = spatial_tokens + self.coord_proj(coords).expand(batch_size, -1, -1)
            query = self.agent_query(agent_tokens)
            spatial_context, _ = self.spatial_attention(query, spatial_tokens, spatial_tokens, need_weights=False)
        # DeepSets aggregation: masked mean is invariant to UAV ordering and
        # stable under padded variable-N batches.
        pooled_agents = masked_mean(agent_tokens, agent_mask, dim=1)
        if spatial_context is not None:
            pooled_spatial_context = masked_mean(spatial_context, agent_mask, dim=1)
            pooled_state_agents = torch.cat([pooled_agents, pooled_spatial_context], dim=-1)
        else:
            pooled_state_agents = pooled_agents
        task_feat = obs["task_id"].reshape(obs["task_id"].shape[0], -1)
        global_feat = obs["global_info"].reshape(obs["global_info"].shape[0], -1)
        state_feat = self.state_mlp(torch.cat([field_feat, pooled_state_agents, task_feat, global_feat], dim=-1))

        batch_size, num_agents, _ = agent_tokens.shape
        expanded_state = state_feat.unsqueeze(1).expand(batch_size, num_agents, -1)
        expanded_task = task_feat.unsqueeze(1).expand(batch_size, num_agents, -1)
        expanded_global = global_feat.unsqueeze(1).expand(batch_size, num_agents, -1)
        # Broadcast global context back to each agent before producing individual
        # waypoint means.
        decoder_parts = [agent_tokens]
        if spatial_context is not None:
            decoder_parts.append(spatial_context)
        decoder_parts += [expanded_state, expanded_task, expanded_global]
        decoder_input = torch.cat(decoder_parts, dim=-1)
        mean = self.decoder(decoder_input)
        if agent_mask is not None:
            mean = mean * agent_mask.unsqueeze(-1).float()
        value = self.value_head(state_feat).squeeze(-1)
        return mean, value

    def get_action_and_value(self, obs: dict[str, torch.Tensor], action: torch.Tensor | None = None):
        """Sample/evaluate tanh-squashed joint actions for PPO-style training."""

        mean, value = self(obs)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp().view(1, 1, 2).expand_as(mean)
        dist = SquashedNormal(mean, std)
        if action is None:
            bounded_action, raw_action = dist.rsample(return_raw=True)
        else:
            if action.ndim == 2:
                action = action.view(mean.shape[0], mean.shape[1], 2)
            bounded_action = torch.clamp(action, -1.0, 1.0)
            raw_action = dist.atanh(bounded_action)
        # First sum x/y dimensions per UAV, then sum valid UAVs to obtain one
        # joint-action log-prob per environment instance.
        log_prob = dist.log_prob(bounded_action, raw_action=raw_action, reduce=False).sum(dim=-1)
        entropy = dist.base_entropy(reduce=False).sum(dim=-1)
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
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        action = SquashedNormal(mean, log_std.exp().view(1, 1, 2).expand_as(mean)).deterministic()
        agent_mask = self._agent_mask(obs)
        if agent_mask is not None:
            action = action * agent_mask.unsqueeze(-1).float()
        return action
