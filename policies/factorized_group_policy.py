from __future__ import annotations

import torch
from torch import nn

from policies.cnn_deepsets_policy import CNNDeepSetsPolicy, masked_mean


class FactorizedGroupPolicy(CNNDeepSetsPolicy):
    """Centralized critic with factorized per-agent actor and soft group tokens.

    The critic still uses one global state embedding. The actor first builds a
    small set of group tokens from the global state, then lets each agent attend
    to those groups before decoding its own waypoint action. This keeps the
    output contract `[B, N, 2]` while introducing an explicit coordination layer
    between "one pooled swarm token" and "independent per-agent heads".
    """

    def __init__(
        self,
        observation_space,
        action_space,
        num_groups: int | None = None,
        group_hidden_dim: int = 128,
        group_assignment_temperature: float = 1.0,
        group_action_strength: float = 0.5,
        use_group_spatial_slots: bool = True,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, **kwargs)
        self.field_hidden_dim = self._infer_field_hidden_dim()
        self.agent_hidden_dim = int(self.agent_encoder[0].out_features)
        self.joint_hidden_dim = int(self.state_mlp[0].out_features)
        self.task_dim = int(observation_space["task_id"].shape[0])
        self.global_dim = int(observation_space["global_info"].shape[0])
        self.spatial_context_dim = self.field_hidden_dim if self.use_spatial_attention else 0
        self.group_count = max(1, min(int(num_groups or self.max_slots), self.max_slots))
        self.group_hidden_dim = int(group_hidden_dim)
        self.group_assignment_temperature = max(float(group_assignment_temperature), 1e-3)
        self.group_action_strength = float(group_action_strength)
        self.use_group_spatial_slots = bool(use_group_spatial_slots)

        self.group_embeddings = nn.Embedding(self.group_count, self.group_hidden_dim)
        self.group_conditioner = nn.Linear(
            self.joint_hidden_dim + self.task_dim + self.global_dim,
            self.group_hidden_dim,
        )
        if self.use_group_spatial_slots and not hasattr(self, "coord_proj"):
            self.coord_proj = nn.Linear(2, self.field_hidden_dim)
        self.agent_group_query = nn.Linear(
            self.agent_hidden_dim + self.spatial_context_dim + self.joint_hidden_dim + self.task_dim + self.global_dim,
            self.group_hidden_dim,
        )
        if self.use_group_spatial_slots:
            self.group_spatial_query = nn.Linear(self.group_hidden_dim, self.field_hidden_dim)
        else:
            self.group_slot_head = nn.Linear(self.group_hidden_dim, 2)
        self.decoder = nn.Sequential(
            nn.Linear(
                self.agent_hidden_dim
                + self.spatial_context_dim
                + self.joint_hidden_dim
                + self.task_dim
                + self.global_dim
                + self.group_hidden_dim,
                int(self.decoder[0].out_features),
            ),
            nn.ReLU(),
            nn.Linear(int(self.decoder[0].out_features), int(self.decoder[2].out_features)),
            nn.ReLU(),
            nn.Linear(int(self.decoder[2].out_features), 2),
        )

    def _infer_field_hidden_dim(self) -> int:
        for module in reversed(list(self.field_encoder.modules())):
            if isinstance(module, nn.Conv2d):
                return int(module.out_channels)
        raise RuntimeError("Failed to infer factorized-group field hidden dim.")

    def _group_tokens(
        self,
        state_feat: torch.Tensor,
        task_feat: torch.Tensor,
        global_feat: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state_feat.shape[0]
        group_ids = torch.arange(self.group_count, device=state_feat.device)
        base_groups = self.group_embeddings(group_ids).unsqueeze(0).expand(batch_size, -1, -1)
        conditioned = self.group_conditioner(torch.cat([state_feat, task_feat, global_feat], dim=-1)).unsqueeze(1)
        return base_groups + conditioned

    def _group_coordinates(
        self,
        group_tokens: torch.Tensor,
        spatial_tokens: torch.Tensor | None,
        spatial_coords: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.use_group_spatial_slots and spatial_tokens is not None and spatial_coords is not None:
            query = self.group_spatial_query(group_tokens)
            logits = torch.matmul(query, spatial_tokens.transpose(1, 2)) / max(spatial_tokens.shape[-1] ** 0.5, 1.0)
            weights = torch.softmax(logits, dim=-1)
            return torch.einsum("bgt,btc->bgc", weights, spatial_coords)
        return torch.tanh(self.group_slot_head(group_tokens))

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        field_map = self.field_encoder(obs["task_field"])
        field_feat = self.field_global_pool(field_map)
        agent_tokens = self.agent_encoder(obs["agents"])
        agent_mask = self._agent_mask(obs)
        if self.use_angular_slot_embeddings and self.slot_embedding_strength > 0.0:
            agent_tokens = agent_tokens + self.slot_embedding_strength * self._angular_slot_features(obs, agent_mask)

        spatial_tokens = None
        spatial_coords = None
        if self.use_spatial_attention or self.use_group_spatial_slots or self.use_spatial_action_head:
            spatial_tokens, spatial_coords = self._build_spatial_tokens(field_map)

        spatial_context = None
        if self.use_spatial_attention:
            query = self.agent_query(agent_tokens)
            spatial_context, _ = self.spatial_attention(query, spatial_tokens, spatial_tokens, need_weights=False)

        pooled_agents = masked_mean(agent_tokens, agent_mask, dim=1)
        pooled_state_agents = pooled_agents
        if spatial_context is not None:
            pooled_state_agents = torch.cat([pooled_agents, masked_mean(spatial_context, agent_mask, dim=1)], dim=-1)
        task_feat = obs["task_id"].reshape(obs["task_id"].shape[0], -1)
        global_feat = obs["global_info"].reshape(obs["global_info"].shape[0], -1)
        state_feat = self.state_mlp(torch.cat([field_feat, pooled_state_agents, task_feat, global_feat], dim=-1))

        batch_size, num_agents, _ = agent_tokens.shape
        expanded_state = state_feat.unsqueeze(1).expand(batch_size, num_agents, -1)
        expanded_task = task_feat.unsqueeze(1).expand(batch_size, num_agents, -1)
        expanded_global = global_feat.unsqueeze(1).expand(batch_size, num_agents, -1)

        group_tokens = self._group_tokens(state_feat, task_feat, global_feat)
        group_coords = self._group_coordinates(group_tokens, spatial_tokens, spatial_coords)
        actor_parts = [agent_tokens]
        if spatial_context is not None:
            actor_parts.append(spatial_context)
        actor_parts += [expanded_state, expanded_task, expanded_global]
        agent_query_input = torch.cat(actor_parts, dim=-1)
        group_query = self.agent_group_query(agent_query_input)
        assignment_logits = torch.matmul(group_query, group_tokens.transpose(1, 2))
        assignment_logits = assignment_logits / (max(self.group_hidden_dim ** 0.5, 1.0) * self.group_assignment_temperature)
        assignment = torch.softmax(assignment_logits, dim=-1)
        assigned_group_context = torch.matmul(assignment, group_tokens)
        assigned_group_coords = torch.matmul(assignment, group_coords)

        decoder_input = torch.cat(actor_parts + [assigned_group_context], dim=-1)
        mean = self.actor_mean_residual_weight * self.decoder(decoder_input)
        positions = obs["agents"][..., :2]
        map_size = torch.clamp(obs["global_info"][..., 4:5], min=1e-6).unsqueeze(1)
        positions_norm = 2.0 * (positions / map_size) - 1.0
        mean = mean + self.group_action_strength * torch.clamp(assigned_group_coords - positions_norm, -2.0, 2.0)

        if self.use_global_spatial_slot_head and self.global_spatial_slot_strength > 0.0:
            mean = mean + self._global_spatial_slot_bias(obs, state_feat, spatial_tokens, spatial_coords, agent_mask)
        if self.use_global_slot_head and self.global_slot_strength > 0.0:
            mean = mean + self._global_slot_bias(obs, state_feat, agent_mask)
        if self.use_coverage_utility_slot_head and self.coverage_utility_slot_strength > 0.0:
            mean = mean + self._coverage_utility_slot_bias(obs, agent_mask)
        if self.use_spatial_action_head and self.spatial_action_strength > 0.0 and spatial_tokens is not None and spatial_coords is not None:
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
