# Task Definitions

## 1. Goal Navigation / Assignment

Goal: multiple UAVs must reach a set of goals with implicit assignment.

Primary task field channels:

- `obstacle`
- `goal_reward`
- `risk`
- dynamic `visited` and `agent_density`

Reward terms:

- positive reward for assignment-weighted distance reduction
- reward for newly reached goals
- penalty for repeated occupation of finished goals
- shared penalties for collisions, path length, time, and safety

Metrics:

- `success`
- `goal_coverage_ratio`
- `completion_time`
- `path_length`
- `collision_rate`

## 2. Coverage / Search

Goal: maximize coverage of demanded area and target discovery probability.

Primary task field channels:

- `obstacle`
- `target_probability`
- `desired_occupancy` as coverage demand
- `visited`
- `agent_density`

Reward terms:

- reward for newly covered demanded cells
- reward for high-probability detection
- penalty for repeated coverage
- shared penalties for time, collisions, and safety

Metrics:

- `coverage_ratio`
- `accumulated_detection_probability`
- `repeated_coverage_ratio`
- `time_discounted_detection_score`
- `spatial_dispersion`

## 3. Formation / Encirclement

Goal: place UAVs into a desired structure around a static or moving target.

Primary task field channels:

- `goal_reward` used as target-location map
- `desired_occupancy`
- `formation_template`
- `obstacle`

Reward terms:

- positive reward for formation-error reduction
- reward for uniform angular coverage
- penalty for radius mismatch
- shared penalties for collisions, path length, and safety

Metrics:

- `formation_error`
- `angular_coverage_uniformity`
- `radius_error`
- `formation_stability`
- `collision_rate`

## 4. Risk-aware Navigation

Goal: solve a goal-navigation task while respecting soft and hard risk structure.

Primary task field channels:

- `obstacle`
- `goal_reward`
- `risk`
- `communication_quality`

Reward terms:

- reward for goal progress
- reward for newly completed goals
- penalty for risk exposure
- penalty for entering no-fly zones
- shared penalties for collisions, path length, and time

Metrics:

- `task_success_rate`
- `goal_coverage_ratio`
- `cumulative_risk_exposure`
- `safety_violation_count`
- `path_length`
