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
        use_spatial_action_head: bool = False,
        spatial_pool_size: int = 8,
        attention_heads: int = 4,
        coordination_repulsion_strength: float = 0.0,
        spatial_action_strength: float = 0.0,
        spatial_target_suppression_strength: float = 0.0,
        spatial_target_suppression_sigma: float = 0.15,
        use_angular_slot_embeddings: bool = False,
        slot_embedding_strength: float = 1.0,
        sector_target_bias_strength: float = 0.0,
        use_global_slot_head: bool = False,
        global_slot_strength: float = 0.0,
        use_global_spatial_slot_head: bool = False,
        global_spatial_slot_strength: float = 0.0,
        use_coverage_utility_slot_head: bool = False,
        coverage_utility_slot_strength: float = 0.0,
        coverage_utility_temperature: float = 8.0,
        coverage_utility_pool_size: int = 16,
        coverage_utility_sector_bias_strength: float = 1.2,
        coverage_utility_suppression_strength: float = 3.0,
        coverage_utility_suppression_sigma: float = 0.25,
        coverage_utility_target_weight: float = 0.6,
        coverage_utility_desired_weight: float = 1.8,
        coverage_utility_visited_weight: float = -0.4,
        coverage_utility_risk_weight: float = -0.2,
        coverage_utility_obstacle_weight: float = -0.8,
        coverage_utility_agent_density_weight: float = -0.3,
        use_coverage_frontier_slot_head: bool = False,
        coverage_frontier_slot_strength: float = 0.0,
        coverage_frontier_temperature: float = 8.0,
        coverage_frontier_pool_size: int = 16,
        coverage_frontier_sector_bias_strength: float = 1.4,
        coverage_frontier_suppression_strength: float = 3.5,
        coverage_frontier_suppression_sigma: float = 0.22,
        coverage_frontier_distance_weight: float = 0.6,
        coverage_frontier_target_weight: float = 0.8,
        coverage_frontier_desired_weight: float = 2.4,
        coverage_frontier_visited_power: float = 1.0,
        coverage_frontier_obstacle_weight: float = -1.2,
        coverage_frontier_agent_density_weight: float = -0.5,
        actor_mean_residual_weight: float = 1.0,
        log_std_min: float = -1.5,
        log_std_max: float = 0.5,
        log_std_init: float = 0.0,
    ):
        super().__init__()
        cnn_channels = cnn_channels or [16, 32, 64]
        self.use_spatial_attention = bool(use_spatial_attention)
        self.use_spatial_action_head = bool(use_spatial_action_head)
        self.spatial_pool_size = int(spatial_pool_size)
        self.coordination_repulsion_strength = float(coordination_repulsion_strength)
        self.spatial_action_strength = float(spatial_action_strength)
        self.spatial_target_suppression_strength = float(spatial_target_suppression_strength)
        self.spatial_target_suppression_sigma = float(spatial_target_suppression_sigma)
        self.use_angular_slot_embeddings = bool(use_angular_slot_embeddings)
        self.slot_embedding_strength = float(slot_embedding_strength)
        self.sector_target_bias_strength = float(sector_target_bias_strength)
        self.use_global_slot_head = bool(use_global_slot_head)
        self.global_slot_strength = float(global_slot_strength)
        self.use_global_spatial_slot_head = bool(use_global_spatial_slot_head)
        self.global_spatial_slot_strength = float(global_spatial_slot_strength)
        self.use_coverage_utility_slot_head = bool(use_coverage_utility_slot_head)
        self.coverage_utility_slot_strength = float(coverage_utility_slot_strength)
        self.coverage_utility_temperature = float(coverage_utility_temperature)
        self.coverage_utility_pool_size = int(coverage_utility_pool_size)
        self.coverage_utility_sector_bias_strength = float(coverage_utility_sector_bias_strength)
        self.coverage_utility_suppression_strength = float(coverage_utility_suppression_strength)
        self.coverage_utility_suppression_sigma = float(coverage_utility_suppression_sigma)
        self.coverage_utility_target_weight = float(coverage_utility_target_weight)
        self.coverage_utility_desired_weight = float(coverage_utility_desired_weight)
        self.coverage_utility_visited_weight = float(coverage_utility_visited_weight)
        self.coverage_utility_risk_weight = float(coverage_utility_risk_weight)
        self.coverage_utility_obstacle_weight = float(coverage_utility_obstacle_weight)
        self.coverage_utility_agent_density_weight = float(coverage_utility_agent_density_weight)
        self.use_coverage_frontier_slot_head = bool(use_coverage_frontier_slot_head)
        self.coverage_frontier_slot_strength = float(coverage_frontier_slot_strength)
        self.coverage_frontier_temperature = float(coverage_frontier_temperature)
        self.coverage_frontier_pool_size = int(coverage_frontier_pool_size)
        self.coverage_frontier_sector_bias_strength = float(coverage_frontier_sector_bias_strength)
        self.coverage_frontier_suppression_strength = float(coverage_frontier_suppression_strength)
        self.coverage_frontier_suppression_sigma = float(coverage_frontier_suppression_sigma)
        self.coverage_frontier_distance_weight = float(coverage_frontier_distance_weight)
        self.coverage_frontier_target_weight = float(coverage_frontier_target_weight)
        self.coverage_frontier_desired_weight = float(coverage_frontier_desired_weight)
        self.coverage_frontier_visited_power = float(coverage_frontier_visited_power)
        self.coverage_frontier_obstacle_weight = float(coverage_frontier_obstacle_weight)
        self.coverage_frontier_agent_density_weight = float(coverage_frontier_agent_density_weight)
        self.actor_mean_residual_weight = float(actor_mean_residual_weight)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.max_slots = int(observation_space["agents"].shape[0])
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
        if self.use_angular_slot_embeddings:
            max_slots = int(observation_space["agents"].shape[0])
            self.angular_slot_embeddings = nn.Embedding(max_slots, agent_hidden_dim)
        spatial_context_dim = 0
        if self.use_spatial_attention or self.use_spatial_action_head:
            self.coord_proj = nn.Linear(2, last_channels)
        if self.use_spatial_attention:
            self.agent_query = nn.Linear(agent_hidden_dim, last_channels)
            self.spatial_attention = nn.MultiheadAttention(
                embed_dim=last_channels,
                num_heads=int(attention_heads),
                batch_first=True,
            )
            spatial_context_dim = last_channels
        task_dim = int(observation_space["task_id"].shape[0])
        global_dim = int(observation_space["global_info"].shape[0])
        if self.use_spatial_action_head:
            self.spatial_action_query = nn.Linear(agent_hidden_dim + joint_hidden_dim + task_dim + global_dim, last_channels)
        # The critic sees a global summary, not individual per-agent decisions.
        # This matches the centralized-agent formulation.
        self.state_mlp = nn.Sequential(
            nn.Linear(last_channels + agent_hidden_dim + spatial_context_dim + task_dim + global_dim, joint_hidden_dim),
            nn.ReLU(),
            nn.Linear(joint_hidden_dim, joint_hidden_dim),
            nn.ReLU(),
        )
        if self.use_global_slot_head:
            self.global_slot_head = nn.Linear(joint_hidden_dim, self.max_slots * 2)
        if self.use_global_spatial_slot_head:
            self.global_spatial_slot_context = nn.Linear(joint_hidden_dim, last_channels)
            self.global_spatial_slot_embeddings = nn.Embedding(self.max_slots, last_channels)
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
        init_log_std = float(min(max(log_std_init, self.log_std_min), self.log_std_max))
        self.log_std = nn.Parameter(torch.full((2,), init_log_std))

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
        if self.use_angular_slot_embeddings and self.slot_embedding_strength > 0.0:
            agent_tokens = agent_tokens + self.slot_embedding_strength * self._angular_slot_features(obs, agent_mask)
        spatial_tokens = None
        spatial_coords = None
        if self.use_spatial_attention or self.use_spatial_action_head:
            spatial_tokens, spatial_coords = self._build_spatial_tokens(field_map)
        spatial_context = None
        if self.use_spatial_attention:
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
        mean = self.actor_mean_residual_weight * self.decoder(decoder_input)
        if self.use_global_spatial_slot_head and self.global_spatial_slot_strength > 0.0:
            mean = mean + self._global_spatial_slot_bias(obs, state_feat, spatial_tokens, spatial_coords, agent_mask)
        if self.use_global_slot_head and self.global_slot_strength > 0.0:
            mean = mean + self._global_slot_bias(obs, state_feat, agent_mask)
        if self.use_coverage_utility_slot_head and self.coverage_utility_slot_strength > 0.0:
            mean = mean + self._coverage_utility_slot_bias(obs, agent_mask)
        if self.use_coverage_frontier_slot_head and self.coverage_frontier_slot_strength > 0.0:
            mean = mean + self._coverage_frontier_slot_bias(obs, agent_mask)
        if self.use_spatial_action_head and self.spatial_action_strength > 0.0:
            mean = mean + self._spatial_action_bias(
                obs,
                agent_tokens,
                expanded_state,
                expanded_task,
                expanded_global,
                spatial_tokens,
                spatial_coords,
                agent_mask,
            )
        if self.coordination_repulsion_strength > 0.0:
            mean = mean + self._coordination_repulsion_bias(obs, agent_mask)
        if agent_mask is not None:
            mean = mean * agent_mask.unsqueeze(-1).float()
        value = self.value_head(state_feat).squeeze(-1)
        return mean, value

    def _build_spatial_tokens(self, field_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled_map = F.adaptive_avg_pool2d(field_map, (self.spatial_pool_size, self.spatial_pool_size))
        batch_size, channels, height, width = pooled_map.shape
        spatial_tokens = pooled_map.flatten(2).transpose(1, 2)
        ys = torch.linspace(-1.0, 1.0, height, device=pooled_map.device, dtype=pooled_map.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=pooled_map.device, dtype=pooled_map.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=-1).reshape(1, height * width, 2).expand(batch_size, -1, -1)
        spatial_tokens = spatial_tokens + self.coord_proj(coords)
        return spatial_tokens, coords

    def _spatial_action_bias(
        self,
        obs: dict[str, torch.Tensor],
        agent_tokens: torch.Tensor,
        expanded_state: torch.Tensor,
        expanded_task: torch.Tensor,
        expanded_global: torch.Tensor,
        spatial_tokens: torch.Tensor,
        spatial_coords: torch.Tensor,
        agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        query_input = torch.cat([agent_tokens, expanded_state, expanded_task, expanded_global], dim=-1)
        query = self.spatial_action_query(query_input)
        logits = torch.matmul(query, spatial_tokens.transpose(1, 2)) / max(spatial_tokens.shape[-1] ** 0.5, 1.0)
        weights = self._spatial_action_weights(logits, spatial_coords, obs, agent_mask)
        target_coords = torch.einsum("bnt,btc->bnc", weights, spatial_coords)

        positions = obs["agents"][..., :2]
        map_size = torch.clamp(obs["global_info"][..., 4:5], min=1e-6).unsqueeze(1)
        positions_norm = 2.0 * (positions / map_size) - 1.0
        delta = torch.clamp(target_coords - positions_norm, -2.0, 2.0)
        if agent_mask is not None:
            delta = delta * agent_mask.unsqueeze(-1).float()
        return self.spatial_action_strength * delta

    def _global_slot_bias(
        self,
        obs: dict[str, torch.Tensor],
        state_feat: torch.Tensor,
        agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        slot_coords = torch.tanh(self.global_slot_head(state_feat)).view(state_feat.shape[0], self.max_slots, 2)
        ranks = self._angular_slot_ranks(obs, agent_mask)
        assigned_slots = torch.gather(slot_coords, 1, ranks.unsqueeze(-1).expand(-1, -1, 2))
        positions = obs["agents"][..., :2]
        map_size = torch.clamp(obs["global_info"][..., 4:5], min=1e-6).unsqueeze(1)
        positions_norm = 2.0 * (positions / map_size) - 1.0
        delta = torch.clamp(assigned_slots - positions_norm, -2.0, 2.0)
        if agent_mask is not None:
            delta = delta * agent_mask.unsqueeze(-1).float()
        return self.global_slot_strength * delta

    def _global_spatial_slot_bias(
        self,
        obs: dict[str, torch.Tensor],
        state_feat: torch.Tensor,
        spatial_tokens: torch.Tensor | None,
        spatial_coords: torch.Tensor | None,
        agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if spatial_tokens is None or spatial_coords is None:
            return torch.zeros_like(obs["agents"][..., :2])
        batch_size = state_feat.shape[0]
        context = self.global_spatial_slot_context(state_feat).unsqueeze(1)
        slot_ids = torch.arange(self.max_slots, device=state_feat.device)
        slot_queries = self.global_spatial_slot_embeddings(slot_ids).unsqueeze(0).expand(batch_size, -1, -1) + context
        logits = torch.matmul(slot_queries, spatial_tokens.transpose(1, 2)) / max(spatial_tokens.shape[-1] ** 0.5, 1.0)
        weights = torch.softmax(logits, dim=-1)
        slot_coords = torch.einsum("bst,btc->bsc", weights, spatial_coords)
        ranks = self._angular_slot_ranks(obs, agent_mask)
        assigned_slots = torch.gather(slot_coords, 1, ranks.unsqueeze(-1).expand(-1, -1, 2))
        positions = obs["agents"][..., :2]
        map_size = torch.clamp(obs["global_info"][..., 4:5], min=1e-6).unsqueeze(1)
        positions_norm = 2.0 * (positions / map_size) - 1.0
        delta = torch.clamp(assigned_slots - positions_norm, -2.0, 2.0)
        if agent_mask is not None:
            delta = delta * agent_mask.unsqueeze(-1).float()
        return self.global_spatial_slot_strength * delta

    def _coverage_utility_slot_bias(
        self,
        obs: dict[str, torch.Tensor],
        agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        task_field = obs["task_field"]
        if task_field.shape[1] < 6:
            return torch.zeros_like(obs["agents"][..., :2])

        obstacle = task_field[:, 0]
        target_probability = task_field[:, 2]
        desired_occupancy = task_field[:, 3]
        risk = task_field[:, 4]
        visited = task_field[:, 5]
        agent_density = task_field[:, 6] if task_field.shape[1] > 6 else torch.zeros_like(visited)
        utility = (
            self.coverage_utility_desired_weight * desired_occupancy
            + self.coverage_utility_target_weight * target_probability
            + self.coverage_utility_visited_weight * visited
            + self.coverage_utility_risk_weight * risk
            + self.coverage_utility_obstacle_weight * obstacle
            + self.coverage_utility_agent_density_weight * agent_density
        )
        if self.coverage_utility_pool_size > 0:
            pool_size = min(int(self.coverage_utility_pool_size), int(utility.shape[-2]), int(utility.shape[-1]))
            utility = F.adaptive_avg_pool2d(utility.unsqueeze(1), (pool_size, pool_size)).squeeze(1)
        batch_size, height, width = utility.shape
        ys = torch.linspace(-1.0, 1.0, height, device=utility.device, dtype=utility.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=utility.device, dtype=utility.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        spatial_coords = torch.stack([grid_x, grid_y], dim=-1).reshape(1, height * width, 2).expand(batch_size, -1, -1)

        logits = utility.reshape(batch_size, 1, height * width).expand(-1, obs["agents"].shape[1], -1)
        logits = self.coverage_utility_temperature * logits
        if self.coverage_utility_sector_bias_strength > 0.0:
            logits = logits + self._coverage_utility_sector_bias(obs, spatial_coords, agent_mask)
        weights = self._coverage_utility_weights(logits, spatial_coords, agent_mask)
        target_coords = torch.einsum("bnt,btc->bnc", weights, spatial_coords)

        positions = obs["agents"][..., :2]
        map_size = torch.clamp(obs["global_info"][..., 4:5], min=1e-6).unsqueeze(1)
        positions_norm = 2.0 * (positions / map_size) - 1.0
        delta = torch.clamp(target_coords - positions_norm, -2.0, 2.0)
        if agent_mask is not None:
            delta = delta * agent_mask.unsqueeze(-1).float()
        return self.coverage_utility_slot_strength * delta

    def _coverage_utility_weights(
        self,
        logits: torch.Tensor,
        spatial_coords: torch.Tensor,
        agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.coverage_utility_suppression_strength <= 0.0:
            weights = torch.softmax(logits, dim=-1)
            if agent_mask is not None:
                weights = weights * agent_mask.unsqueeze(-1).float()
            return weights

        coords_rel = spatial_coords.unsqueeze(2) - spatial_coords.unsqueeze(1)
        dist_sq = (coords_rel ** 2).sum(dim=-1)
        sigma = max(self.coverage_utility_suppression_sigma, 1e-6)
        suppression_kernel = torch.exp(-dist_sq / sigma)

        batch_size, num_agents, num_tokens = logits.shape
        weights = torch.zeros_like(logits)
        suppression = torch.zeros((batch_size, num_tokens), device=logits.device, dtype=logits.dtype)
        for agent_idx in range(num_agents):
            adjusted_logits = logits[:, agent_idx, :] - self.coverage_utility_suppression_strength * suppression
            current_weights = torch.softmax(adjusted_logits, dim=-1)
            if agent_mask is not None:
                valid = agent_mask[:, agent_idx].unsqueeze(-1).float()
                current_weights = current_weights * valid
            weights[:, agent_idx, :] = current_weights
            suppression = suppression + torch.einsum("bt,bts->bs", current_weights, suppression_kernel)
        return weights

    def _coverage_utility_sector_bias(
        self,
        obs: dict[str, torch.Tensor],
        spatial_coords: torch.Tensor,
        agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        ranks = self._angular_slot_ranks(obs, agent_mask)
        num_agents = obs["agents"].shape[1]
        sector_centers = (2.0 * torch.pi) * (ranks.float() + 0.5) / max(num_agents, 1)
        token_angles = torch.atan2(spatial_coords[..., 1], spatial_coords[..., 0]).unsqueeze(1)
        sector_centers = sector_centers.unsqueeze(-1)
        angle_delta = torch.atan2(
            torch.sin(token_angles - sector_centers),
            torch.cos(token_angles - sector_centers),
        ).abs()
        bias = self.coverage_utility_sector_bias_strength * (1.0 - angle_delta / torch.pi)
        if agent_mask is not None:
            bias = bias * agent_mask.unsqueeze(-1).float()
        return bias

    def _coverage_frontier_slot_bias(
        self,
        obs: dict[str, torch.Tensor],
        agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        task_field = obs["task_field"]
        if task_field.shape[1] < 7:
            return torch.zeros_like(obs["agents"][..., :2])

        obstacle = task_field[:, 0]
        target_probability = task_field[:, 2]
        desired_occupancy = task_field[:, 3]
        visited = task_field[:, 5]
        agent_density = task_field[:, 6]
        unvisited = torch.clamp(1.0 - visited, 0.0, 1.0) ** max(self.coverage_frontier_visited_power, 1e-6)
        remaining_demand = desired_occupancy * unvisited
        utility = (
            self.coverage_frontier_desired_weight * remaining_demand
            + self.coverage_frontier_target_weight * target_probability * unvisited
            + self.coverage_frontier_obstacle_weight * obstacle
            + self.coverage_frontier_agent_density_weight * agent_density
        )
        if self.coverage_frontier_pool_size > 0:
            pool_size = min(int(self.coverage_frontier_pool_size), int(utility.shape[-2]), int(utility.shape[-1]))
            utility = F.adaptive_avg_pool2d(utility.unsqueeze(1), (pool_size, pool_size)).squeeze(1)

        batch_size, height, width = utility.shape
        ys = torch.linspace(-1.0, 1.0, height, device=utility.device, dtype=utility.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=utility.device, dtype=utility.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        spatial_coords = torch.stack([grid_x, grid_y], dim=-1).reshape(1, height * width, 2).expand(batch_size, -1, -1)

        num_agents = obs["agents"].shape[1]
        logits = self.coverage_frontier_temperature * utility.reshape(batch_size, 1, height * width).expand(-1, num_agents, -1)
        if self.coverage_frontier_sector_bias_strength > 0.0:
            logits = logits + self._coverage_frontier_sector_bias(obs, spatial_coords, agent_mask)
        if self.coverage_frontier_distance_weight > 0.0:
            positions = obs["agents"][..., :2]
            map_size = torch.clamp(obs["global_info"][..., 4:5], min=1e-6).unsqueeze(1)
            positions_norm = 2.0 * (positions / map_size) - 1.0
            distance = torch.linalg.norm(spatial_coords.unsqueeze(1) - positions_norm.unsqueeze(2), dim=-1)
            logits = logits - self.coverage_frontier_distance_weight * distance

        weights = self._coverage_frontier_weights(logits, spatial_coords, agent_mask)
        target_coords = torch.einsum("bnt,btc->bnc", weights, spatial_coords)

        positions = obs["agents"][..., :2]
        map_size = torch.clamp(obs["global_info"][..., 4:5], min=1e-6).unsqueeze(1)
        positions_norm = 2.0 * (positions / map_size) - 1.0
        delta = torch.clamp(target_coords - positions_norm, -2.0, 2.0)
        if agent_mask is not None:
            delta = delta * agent_mask.unsqueeze(-1).float()
        return self.coverage_frontier_slot_strength * delta

    def _coverage_frontier_weights(
        self,
        logits: torch.Tensor,
        spatial_coords: torch.Tensor,
        agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.coverage_frontier_suppression_strength <= 0.0:
            weights = torch.softmax(logits, dim=-1)
            if agent_mask is not None:
                weights = weights * agent_mask.unsqueeze(-1).float()
            return weights

        coords_rel = spatial_coords.unsqueeze(2) - spatial_coords.unsqueeze(1)
        dist_sq = (coords_rel ** 2).sum(dim=-1)
        sigma = max(self.coverage_frontier_suppression_sigma, 1e-6)
        suppression_kernel = torch.exp(-dist_sq / sigma)

        batch_size, num_agents, num_tokens = logits.shape
        weights = torch.zeros_like(logits)
        suppression = torch.zeros((batch_size, num_tokens), device=logits.device, dtype=logits.dtype)
        for agent_idx in range(num_agents):
            adjusted_logits = logits[:, agent_idx, :] - self.coverage_frontier_suppression_strength * suppression
            current_weights = torch.softmax(adjusted_logits, dim=-1)
            if agent_mask is not None:
                current_weights = current_weights * agent_mask[:, agent_idx].unsqueeze(-1).float()
            weights[:, agent_idx, :] = current_weights
            suppression = suppression + torch.einsum("bt,bts->bs", current_weights, suppression_kernel)
        return weights

    def _coverage_frontier_sector_bias(
        self,
        obs: dict[str, torch.Tensor],
        spatial_coords: torch.Tensor,
        agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        ranks = self._angular_slot_ranks(obs, agent_mask)
        num_agents = obs["agents"].shape[1]
        sector_centers = (2.0 * torch.pi) * (ranks.float() + 0.5) / max(num_agents, 1)
        token_angles = torch.atan2(spatial_coords[..., 1], spatial_coords[..., 0]).unsqueeze(1)
        sector_centers = sector_centers.unsqueeze(-1)
        angle_delta = torch.atan2(
            torch.sin(token_angles - sector_centers),
            torch.cos(token_angles - sector_centers),
        ).abs()
        bias = self.coverage_frontier_sector_bias_strength * (1.0 - angle_delta / torch.pi)
        if agent_mask is not None:
            bias = bias * agent_mask.unsqueeze(-1).float()
        return bias

    def _spatial_action_weights(
        self,
        logits: torch.Tensor,
        spatial_coords: torch.Tensor,
        obs: dict[str, torch.Tensor],
        agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.sector_target_bias_strength > 0.0:
            logits = logits + self._sector_target_bias(obs, spatial_coords, agent_mask)
        if self.spatial_target_suppression_strength <= 0.0:
            return torch.softmax(logits, dim=-1)

        coords_rel = spatial_coords.unsqueeze(2) - spatial_coords.unsqueeze(1)
        dist_sq = (coords_rel ** 2).sum(dim=-1)
        suppression_kernel = torch.exp(-dist_sq / max(self.spatial_target_suppression_sigma, 1e-6))

        batch_size, num_agents, num_tokens = logits.shape
        weights = torch.zeros_like(logits)
        suppression = torch.zeros((batch_size, num_tokens), device=logits.device, dtype=logits.dtype)
        for agent_idx in range(num_agents):
            adjusted_logits = logits[:, agent_idx, :] - self.spatial_target_suppression_strength * suppression
            current_weights = torch.softmax(adjusted_logits, dim=-1)
            if agent_mask is not None:
                valid = agent_mask[:, agent_idx].unsqueeze(-1).float()
                current_weights = current_weights * valid
            weights[:, agent_idx, :] = current_weights
            suppression = suppression + torch.einsum("bt,bts->bs", current_weights, suppression_kernel)
        return weights

    def _sector_target_bias(
        self,
        obs: dict[str, torch.Tensor],
        spatial_coords: torch.Tensor,
        agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        ranks = self._angular_slot_ranks(obs, agent_mask)
        num_agents = obs["agents"].shape[1]
        sector_centers = (2.0 * torch.pi) * (ranks.float() + 0.5) / max(num_agents, 1)
        token_angles = torch.atan2(spatial_coords[..., 1], spatial_coords[..., 0]).unsqueeze(1)
        sector_centers = sector_centers.unsqueeze(-1)
        angle_delta = torch.atan2(
            torch.sin(token_angles - sector_centers),
            torch.cos(token_angles - sector_centers),
        ).abs()
        bias = self.sector_target_bias_strength * (1.0 - angle_delta / torch.pi)
        if agent_mask is not None:
            bias = bias * agent_mask.unsqueeze(-1).float()
        return bias

    def _angular_slot_ranks(
        self,
        obs: dict[str, torch.Tensor],
        agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        positions = obs["agents"][..., :2]
        if agent_mask is None:
            centroid = positions.mean(dim=1, keepdim=True)
        else:
            centroid = masked_mean(positions, agent_mask, dim=1).unsqueeze(1)
        rel = positions - centroid
        angles = torch.atan2(rel[..., 1], rel[..., 0])
        if agent_mask is not None:
            angles = torch.where(agent_mask, angles, torch.full_like(angles, 1e9))
        order = torch.argsort(angles, dim=1)
        ranks = torch.argsort(order, dim=1)
        if agent_mask is not None:
            ranks = torch.where(agent_mask, ranks, torch.zeros_like(ranks))
        return ranks

    def _angular_slot_features(
        self,
        obs: dict[str, torch.Tensor],
        agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        ranks = self._angular_slot_ranks(obs, agent_mask)
        return self.angular_slot_embeddings(ranks)

    def _coordination_repulsion_bias(
        self,
        obs: dict[str, torch.Tensor],
        agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Add a small learned-free repulsion bias to encourage agent spread.

        This is disabled by default and only intended for coverage-style
        experiments where repeated overlap dominates before the policy learns to
        partition the map.
        """

        positions = obs["agents"][..., :2]
        map_size = torch.clamp(obs["global_info"][..., 4:5], min=1e-6).unsqueeze(1)
        positions_norm = positions / map_size
        rel = positions_norm.unsqueeze(2) - positions_norm.unsqueeze(1)
        dist_sq = (rel ** 2).sum(dim=-1, keepdim=True)

        batch_size, num_agents, _, _ = rel.shape
        eye = torch.eye(num_agents, device=rel.device, dtype=torch.bool).view(1, num_agents, num_agents, 1)
        valid_pairs = ~eye
        if agent_mask is not None:
            pair_mask = (agent_mask.unsqueeze(2) & agent_mask.unsqueeze(1)).unsqueeze(-1)
            valid_pairs = valid_pairs & pair_mask

        weighted_rel = torch.where(valid_pairs, rel / torch.clamp(dist_sq, min=1e-4), torch.zeros_like(rel))
        repulsion = weighted_rel.sum(dim=2)
        repulsion_norm = torch.linalg.norm(repulsion, dim=-1, keepdim=True)
        repulsion_dir = repulsion / torch.clamp(repulsion_norm, min=1e-6)
        return self.coordination_repulsion_strength * repulsion_dir

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
