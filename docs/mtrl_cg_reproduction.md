# MTRL-CG Reproduction Notes

Paper: `MTRL-CG: Multi-Task Reinforcement Learning Method with Spectral Clustering-Based Task Grouping`

Official code: <https://github.com/zhangt603/MTRL-CG.git>

## Paper Summary

MTRL-CG addresses negative interference in multi-task reinforcement learning. It first trains a shared SAC-style critic/policy, estimates how a task-specific critic update changes Q-values on other tasks, constructs an inter-task affinity matrix, applies signed spectral clustering, and then trains a separate multi-task SAC learner per task group.

Core pipeline:

1. Train a shared multi-task SAC backbone for `Tg` timesteps.
2. Every `C` timesteps, estimate directed task affinity `W_ij` from the effect of task `i`'s critic update on task `j`'s Q-values.
3. Sum affinity estimates, symmetrize `W`, split positive/negative edges, and solve the signed generalized eigenproblem from SPONGE.
4. Cluster task spectral embeddings with k-means++.
5. Train one dedicated SAC policy and critic per task group for `Tm` timesteps.

The paper evaluates on Meta-World MT10 and MT50, fixed and mixed settings, using success rate. It reports gains when wrapping MT-SAC, CARE, CMTA, and PaCo with MTRL-CG. The authors use three seeds and find three groups to work best on MT10 in their ablation.

## Reproduction Difficulty

Difficulty is medium-high for exact paper reproduction:

- Meta-World MT10/MT50 long-budget runs are compute-heavy.
- The paper does not list all low-level hyperparameters for `Tg`, `C`, `Tm`, replay sizes, or exact backbone configs in the PDF.
- Official code exists, but it is organized as modified copies of multiple research codebases rather than a single polished package.
- Exact affinity values depend on critic implementation details, seeds, replay contents, and Q-value scaling.

Difficulty is medium for a faithful method reproduction:

- The algorithm is conceptually simple once SAC and per-task replay/evaluation exist.
- Wayffusion already has multi-task SAC and fixed-task evaluation, so the core grouping idea can be implemented directly.
- The first Wayffusion implementation should be treated as a method-level reproduction, not a claim of matching Meta-World numbers.

## Wayffusion Implementation

Estimate affinity and group tasks:

```bash
/opt/conda/bin/python scripts/mtrl_cg_estimate_wayffusion_affinity.py \
  --config configs/policy/sac_server_smoke.yaml \
  --tasks goal_nav coverage formation risk_nav \
  --agent_counts 4 \
  --pretrain_steps 64 \
  --estimate_interval 16 \
  --affinity_batch 16 \
  --num_groups 2
```

Outputs:

```text
outputs/mtrl_cg/<task_set>_N<agent_count>/
  affinity_raw.csv
  affinity_symmetric.csv
  groups.yaml
  metadata.yaml
```

Train group-wise SAC from the grouping file:

```bash
/opt/conda/bin/python scripts/train_mtrl_cg_wayffusion_groups.py \
  --groups outputs/mtrl_cg/goal_nav_coverage_formation_risk_nav_N4/groups.yaml \
  --config configs/policy/sac_cnn_deepsets.yaml \
  --agent_counts 4
```

## Public Benchmark Route

For Meta-World reproduction, start from the official repository:

```bash
git clone https://github.com/zhangt603/MTRL-CG.git external/MTRL-CG
```

Then follow the official repository layout:

- `Affinity/`: run modified backbone training to produce affinity matrices.
- `group_method/group.py`: run signed spectral clustering on the saved matrix.
- `group-wise/`: run group-wise training for MT-SAC, CARE, CMTA, or PaCo.

Recommended first target:

1. MT10-Fixed with MT-SAC only.
2. Three seeds after a one-seed smoke succeeds.
3. Compare MT-SAC vs MT-SAC+MTRL-CG max/final success rate.

