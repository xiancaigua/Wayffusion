# Evaluation Protocol

## Experiment 0: Task field sanity check

Command:

```powershell
python scripts/check/generate_task_fields.py --config configs/env/multitask.yaml
```

What it does:

- samples one representative episode per task
- renders the full task field channels
- runs the heuristic waypoint policy
- saves rollout plots and `metrics.csv`

Outputs:

- `outputs/smoke/sanity/task_fields/*.png`
- `outputs/smoke/sanity/trajectories/*.png`
- `outputs/smoke/sanity/metrics.csv`

## Experiment 1: Single-task baseline

Command:

```powershell
python scripts/evaluate_baselines.py --config configs/eval/eval_single_task.yaml
```

Protocol:

- evaluate `heuristic` and `random` on each of the four tasks
- default 20 episodes per `(task, policy)` pair
- save best rollout plot per pair

Primary metrics:

- task success
- episode reward
- collision count
- path length
- task-specific metrics

## Experiment 2: Multi-task sanity benchmark

Command:

```powershell
python scripts/evaluate_baselines.py --config configs/eval/eval_multitask.yaml
```

Protocol:

- task sampled from the environment distribution every episode
- compare `heuristic` vs `random`
- report per-episode normalized score and task labels

Suggested summary aggregation:

- average normalized score across all episodes
- worst-task average score
- per-task average score

## Experiment 3: Generalization split

Command:

```powershell
python scripts/evaluate_baselines.py --config configs/eval/eval_generalization.yaml
```

Train-style distribution:

- `num_agents = 4`
- lower obstacle density
- static formation target
- simple coverage fields
- seen formation templates

Held-out distribution:

- `num_agents = 6`
- higher obstacle density
- linear target motion
- multi-peak coverage fields
- held-out formation template family

## PPO evaluation

Train:

```powershell
python scripts/train_ppo.py --config configs/policy/ppo_mlp.yaml --env-config configs/env/multitask.yaml --tasks goal_nav coverage --agent_counts 4
```

Evaluate:

```powershell
python scripts/evaluate_policy.py --checkpoint outputs/training/ppo/<timestamp>/ppo_mlp_goal_nav_coverage_N4_multi_channel_field_plus_task_id/checkpoints/checkpoint_0020.pt --policy-config configs/policy/ppo_mlp.yaml --tasks goal_nav coverage --agent_counts 4
python scripts/evaluate_policy.py --checkpoint outputs/training/ppo/<timestamp>/ppo_mlp_goal_nav_coverage_N4_multi_channel_field_plus_task_id/checkpoints/checkpoint_0020.pt --policy-config configs/policy/ppo_mlp.yaml --tasks goal_nav coverage --agent_counts 4 8 10
```
