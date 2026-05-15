# Algorithm Comparison

The benchmark now supports centralized learning-baseline comparison across:

- `random`
- `heuristic`
- `BC`
- `PPO`
- `BC + PPO`
- `SAC`
- `TD3`

## Core comparison command

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_algorithms.py --configs heuristic random bc ppo bc_ppo sac td3 --tasks goal_nav coverage --agent_counts 4 8 10 --episodes 4 --output-path outputs/eval/algorithm_comparison.csv
```

## Interpretation

- `heuristic` is the reference expert-like controller and is also used for expert dataset generation.
- `random` is the lower-bound reference.
- `normalized_score` is computed against the random and heuristic returns for the same task set, scaling mode, and swarm size.
- `BC + PPO` is the main practical learning baseline because it combines expert warm-start with centralized policy improvement.
- `SAC` and `TD3` are useful to test whether off-policy continuous-control training is more sample-efficient than PPO under a joint waypoint action.

## Current practical advice

- Use `CNN + DeepSets` for variable-agent experiments.
- Use `MLP` only for fixed-`N` small-scale controls.
- Use `BC + PPO` or `SAC` first when the PPO smoke baseline is too weak.
