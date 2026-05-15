# Large-N Calibration Round 2 Plan

## Goal
Stabilize large-N reference normalization and coverage/goal difficulty scaling, then retrain BC+PPO as the main density-preserving learning baseline under the recalibrated contract.

## User Constraints
- Keep the benchmark centralized single-policy; no MARL or per-agent policies.
- Work inside the dedicated workspace virtual environment only.
- Prioritize large-N calibration first, then BC+PPO retraining.

## Main Questions
1. Why does large-N normalized score become unstable under density-preserving evaluation?
2. Which difficulty-scaling terms still allow trivial or misleading success/return behavior?
3. After recalibration, does BC+PPO improve on N={20,40} and remain interpretable on N={80,100}?

## Scope
- Code touchpoints: reward normalization, task difficulty scaling, evaluation normalization, docs, and summary outputs.
- Main training line: BC+PPO, CNN+DeepSets, density-preserving, goal_nav + coverage, train N={20,40}.
- Validation: targeted calibration smokes, eval tables, and test suite.

## Planned Steps
1. Audit the current normalization formula, reference estimation path, and large-N task difficulty rules.
2. Implement reference-gap stabilization and any remaining difficulty calibration fixes.
3. Update docs and summary notes for the revised contract.
4. Retrain BC+PPO with the revised setup.
5. Re-evaluate cross-N behavior and compare against the prior run.
6. Run tests and summarize recommendations.

## Acceptance Signals
- Large-N reference normalization no longer produces obviously explosive scores from tiny heuristic-random gaps.
- Coverage and goal difficulty scale more smoothly at N>=20.
- BC+PPO retrain completes and produces fresh eval artifacts under the new calibration.
- Tests pass.

## Risks
- Large-N eval is slow, especially N=80/100.
- Tightening calibration may lower absolute scores before improving stability.
- SAC/other baselines are out of scope unless needed for spot-checking.

## Outputs
- Updated code and docs.
- New BC+PPO training directory and eval CSVs.
- Updated stage summary with calibration and retraining takeaways.
