# Learning Baselines

This stage hardens the benchmark from a rollout sanity benchmark into a learning-baseline benchmark. The setup remains centralized single-agent RL: one policy receives the global task field, all agent states, and the task identifier, then emits the joint waypoint action for the whole swarm.

## Supported algorithms

- `PPO`: on-policy baseline with reward normalization, advantage normalization, gradient clipping, entropy regularization, configurable rollout length, mini-batches, synchronous vectorized rollout collection, and optional linear learning-rate decay.
- `SAC`: centralized continuous-control baseline with a joint action of size `2N`, twin Q-functions, replay buffer, and tanh-squashed actor outputs.
- `TD3`: centralized deterministic continuous-control baseline with twin critics, delayed actor updates, target policy smoothing, and replay buffer.
- `BC`: supervised imitation using heuristic waypoint actions as expert targets.
- `BC + PPO`: initialize the centralized policy from a BC checkpoint, then fine-tune with PPO.
- `BC + SAC`: configuration skeleton is provided for future actor initialization from BC.

## Policy architectures

- `MLP`: flatten the whole observation and output the full joint action. This is only intended for fixed-`N` small-scale settings.
- `CNN + DeepSets`: the main variable-`N` architecture. The task field is encoded with a CNN, each agent state is encoded with a shared MLP, pooled swarm context is aggregated with DeepSets-style permutation-aware pooling, and the decoder outputs a waypoint per agent.
- `CNN + Attention`: a transformer-style variable-`N` encoder where the field context and task embedding are fused with agent tokens before per-agent decoding.

## Observation modes

All learning baselines use the same environment API and the same shared observation/action structure. Observation ablations are implemented through adapters rather than task-specific environment variants:

- `task_id_only`
- `single_channel_field`
- `multi_channel_field`
- `multi_channel_field + task_id`
- `multi_channel_field + agent_density_map`
- `multi_channel_field without risk channel`
- `multi_channel_field without desired_occupancy channel`

All training and evaluation scripts expose these through `--obs_variant`, which makes representation ablations easy to reproduce without maintaining separate environment files.

## Recommended starting point

For the current benchmark stage, the most useful paper-facing starting point is:

- algorithm: `BC + PPO`
- architecture: `CNN + DeepSets`
- tasks: `goal_nav + coverage`
- agent counts: `4, 8, 10`
- scaling mode: `fixed_map` first, then `density_preserving`

This path gives a stronger warm start than PPO-from-scratch while preserving the centralized RL formulation.

For large-`N` density-preserving evaluation, use the recalibrated reference contract from `reward_normalization.md`: normalize per task family, not from one mixed-task return, and inspect `reference_order_flipped` whenever heuristic and random swap rank on a hard slice.
