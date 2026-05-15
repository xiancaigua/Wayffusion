# Scaling Experiments

## Agent-count settings

The environment and evaluation stack support the following swarm sizes:

- small scale: `N = 4, 8, 10`
- medium scale: `N = 20, 40`
- stress test: `N = 80, 100`

Configs are provided in:

- `configs/env/agents_4.yaml`
- `configs/env/agents_8.yaml`
- `configs/env/agents_10.yaml`
- `configs/env/agents_20.yaml`
- `configs/env/agents_40.yaml`
- `configs/env/agents_80.yaml`
- `configs/env/agents_100.yaml`
- `configs/env/scaling.yaml`

## Protocol A: Fixed-N training and evaluation

Train one centralized policy per swarm size and evaluate it only on the same `N`.

Example:

```powershell
.\.venv\Scripts\python.exe scripts/train_ppo.py --config configs/policy/ppo_mlp.yaml --tasks goal_nav coverage --agent_counts 4
.\.venv\Scripts\python.exe scripts/evaluate_scaling.py --checkpoint outputs/training/ppo/<timestamp>/ppo_mlp_goal_nav_coverage_N4/checkpoints/checkpoint_0008.pt --policy-config configs/policy/ppo_mlp.yaml --tasks goal_nav coverage --agent_counts 4 --protocol fixed_N --train_agent_counts 4 --output-path outputs/eval/scaling_fixed_N.csv
```

## Protocol B: Variable-N training and cross-N evaluation

Train one variable-agent centralized policy on a set of swarm sizes, then evaluate it across the full range.

Recommended train sets:

- `small_train = {4, 8, 10}`
- `medium_train = {4, 8, 10, 20, 40}`
- `large_train = {4, 8, 10, 20, 40, 80, 100}`

Example:

```powershell
.\.venv\Scripts\python.exe scripts/train_ppo.py --config configs/policy/ppo_cnn_deepsets.yaml --tasks goal_nav coverage --agent_counts 4 8 10
.\.venv\Scripts\python.exe scripts/evaluate_scaling.py --checkpoint outputs/training/ppo/<timestamp>/ppo_cnn_deepsets_goal_nav_coverage_N4_8_10/checkpoints/checkpoint_0008.pt --policy-config configs/policy/ppo_cnn_deepsets.yaml --tasks goal_nav coverage --agent_counts 4 8 10 20 40 80 100 --protocol variable_N --train_agent_counts 4 8 10 --output-path outputs/eval/scaling_variable_N.csv
```

## Scaling modes

Two scaling modes are implemented.

- `fixed_map`: the map remains fixed, so larger `N` increases crowding and collision pressure.
- `density_preserving`: map size and task scale grow with `N`, which keeps agent density more stable and better measures algorithmic scalability.

In the current implementation, task scale also grows with `N` even under `fixed_map`. This avoids trivial large-swarm success caused by keeping the same small goal set or single-pass coverage demand while dramatically increasing swarm size.
For `density_preserving`, goal counts and coverage peak counts now use a smoother sublinear exponent, while coverage revisit requirements still increase with `N`. This keeps large-`N` tasks challenging without making every density-preserving slice collapse into the same zero-signal regime.

## Reported metrics

Every scaling evaluation row includes:

- return mean/std
- normalized score
- success rate
- collision rate
- path length
- task-specific metric columns when available
- inference latency
- rollout steps per second
- wall-clock time
- process memory usage

For cross-`N` evaluation, `evaluate_scaling.py` also attaches:

- `train_agent_set`
- `average_normalized_score`
- `worst_task_score`

For coverage, large-`N` runs use multi-visit fulfillment in both scaling modes. `fixed_map` uses a stronger revisit requirement because crowding is the point of the protocol, while `density_preserving` still increases the visit requirement mildly so large swarms do not solve coverage trivially from one fast sweep.
