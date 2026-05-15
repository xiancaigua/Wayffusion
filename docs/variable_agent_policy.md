# Variable-Agent Policy

## Why a variable-agent policy is needed

The benchmark treats the whole swarm as one centralized agent, but the swarm size `N` may change across experiments. A flatten-MLP policy is tied to a fixed input size and therefore cannot serve as the main architecture for cross-`N` experiments.

## Shared formulation

The formulation stays centralized:

```text
pi_theta(A | task_field, agents, task_id)
```

with:

- `task_field`: `[C, H, W]`
- `agents`: `[N, d_agent]`
- `task_id`: `[d_task]`
- action `A`: `[N, 2]`

## Implemented variable-N architectures

### CNN + DeepSets

- encodes the field with a CNN
- encodes each agent with a shared MLP
- aggregates pooled swarm context with permutation-aware set pooling
- decodes a waypoint per agent

This is the default variable-`N` backbone.

### CNN + Attention

- converts field context into a global token
- embeds agent states as agent tokens
- mixes them through multi-head self-attention
- decodes per-agent waypoints from updated agent tokens

This is useful when interactions become more structured at larger `N`.

## Padding and masks

Mixed-`N` expert datasets are stored with padding up to the largest `N` in the dataset. An `agent_mask` marks valid agent entries so the policy loss is computed only on active agents.

## Practical guideline

- use `MLP` for fixed-`N`, small-scale sanity experiments
- use `CNN + DeepSets` for the main scaling benchmark
- use `CNN + Attention` when testing whether richer agent-agent interaction modeling helps at medium or large `N`
