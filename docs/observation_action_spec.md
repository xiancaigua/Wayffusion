# Observation and Action Spec

## Observation

The environment returns a Gymnasium-style observation dictionary:

```python
{
  "task_field": np.ndarray,
  "agents": np.ndarray,
  "task_id": np.ndarray,
  "global_info": np.ndarray,
}
```

### `task_field`

- Default shape: `[9, H, W]`
- Single-channel ablation: `[1, H, W]`
- `no_spatial_field`: same default shape, but all zeros
- `task_id_only`: deprecated alias for `no_spatial_field`

Fixed channel order:

1. `obstacle`
2. `goal_reward`
3. `target_probability`
4. `desired_occupancy`
5. `risk`
6. `visited`
7. `agent_density`
8. `communication_quality`
9. `formation_template`

### `agents`

- Shape: `[N, 6]`
- Per-agent layout:
  - `x`
  - `y`
  - `vx`
  - `vy`
  - `battery`
  - `role_id`

Ranges:

- positions in `[0, map_size]`
- velocities clipped by `max_speed`
- battery is currently fixed at `1`
- role id is currently `0`

### `task_id`

- Shape: `[4]`
- One-hot order:
  - `goal_nav`
  - `coverage`
  - `formation`
  - `risk_nav`

### `global_info`

- Shape: `[5]`
- Layout:
  - normalized step progress
  - average collision rate so far
  - average risk exposure so far
  - task completion proxy
  - current `map_size`

## Action

- Shape: `[N, 2]`
- Range: `[-1, 1]`
- Semantics: relative waypoint delta for each UAV

Execution:

```python
delta = action * max_waypoint_step
g_i = clip(p_i + delta_i, 0, map_size)
v_i = clip(k_p * (g_i - p_i), max_speed)
p_i_next = clip(p_i + dt * v_i, 0, map_size)
```

The policy never emits low-level motor commands or direct task-allocation labels.
