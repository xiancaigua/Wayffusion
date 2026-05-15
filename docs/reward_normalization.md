# Reward Normalization

## Motivation

Raw returns vary a lot across tasks, swarm sizes, and scaling modes. A larger swarm can accumulate more path cost, more collision cost, or more coverage reward, so raw return alone is not a stable comparison target.

## Diagnostics

Use:

```powershell
.\.venv\Scripts\python.exe scripts/check/diagnose_rewards.py --tasks all --agent_counts 4 8 10 20 40 80 100
```

This produces:

- `outputs/smoke/diagnostics/reward_stats.csv`
- `outputs/smoke/diagnostics/reward_components/`

Each row reports:

- total return mean/std
- success rate
- reward-component mean/std
- collision penalties
- path penalties
- task-specific reward terms

## Training-time normalization

The PPO trainer uses:

- running reward normalization
- advantage normalization
- gradient clipping

These changes are intended to improve training stability without changing the centralized environment definition.

At the environment level, large-scale reward calibration now also uses scale-aware component accounting:

- collision and obstacle penalties are normalized by agent count
- path effort is normalized by agent count and one-step motion scale
- risk and safety penalties are normalized by agent count
- time penalty is scaled against spatial growth in `density_preserving`
- coverage uses fulfillment-based accounting with scale-dependent revisit requirements instead of binary first-touch success
- density-preserving goal counts and coverage peak counts now grow with a smoother sublinear exponent before the revisit constraint is applied

This keeps reward magnitudes more comparable across `N`.

## Evaluation-time normalized score

We report a stabilized reference-normalized score derived from the same two non-learning anchors:

```text
S_raw = (R - R_low) / (R_high - R_low + eps)
```

where:

- `R` is the evaluated policy return
- `R_low` and `R_high` are the lower and upper returns from `{R_random, R_heuristic}` on the same task/scaling setup

To make large-`N` multi-task tables more stable, the benchmark now:

- computes references per task family first, then averages over the actually sampled episode tasks
- sorts `{R_random, R_heuristic}` into lower and upper anchors before normalization, so slices remain monotonic even when the heuristic temporarily underperforms random
- clips the stored `normalized_score` to `[-5, 5]` while keeping the raw ratio as `normalized_score_raw` during episode aggregation

The score can therefore be:

- below `0` if the policy underperforms random
- around `1` if the policy matches heuristic
- above `1` if the policy outperforms the current heuristic baseline

If the heuristic is actually weaker than random on a difficult slice, `reference_order_flipped = 1` marks that the random baseline became the stronger anchor for that task/scale pair.

## Metric calibration notes

For cross-`N` comparisons, the benchmark now reports:

- `collision_rate` as average collision count per agent-step
- `path_length` as per-agent cumulative path length
- `risk_exposure` as per-agent cumulative risk exposure

Raw totals are still kept when useful, but the per-agent metrics are the primary scaling-facing comparisons.
