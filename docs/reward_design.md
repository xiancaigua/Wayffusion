# Reward Design

## Shared environment penalties

Every task receives the task-specific reward plus shared penalties:

```text
R_total = R_task
        + w_collision * pair_collision_count
        + w_obstacle * obstacle_collision_count
        + w_path * path_length_delta
        + w_time
        + w_safety * safety_violations
        + w_risk * step_risk_exposure
```

Default shared weights are defined in `configs/env/base.yaml`.

## Goal navigation

```text
R_goal = w_progress * (prev_assignment_cost - current_assignment_cost)
       + w_reached * newly_reached_goals
       + w_repeat * repeated_goal_occupancy
```

Intent:

- reward moving the team toward good implicit assignments
- strongly reward actual completion
- discourage multiple agents wasting time on already-completed goals

## Coverage

```text
R_cov = w_new * new_coverage_ratio
      + w_prob * high_probability_detection_gain
      + w_repeat * repeated_coverage_ratio
```

Intent:

- prioritize unseen demanded area
- favor high-probability search zones
- discourage redundant revisits

## Formation

```text
R_form = w_error * formation_error_reduction
       + w_angle * angular_coverage_uniformity
       - w_radius_penalty * radius_error
       + w_stability * short_horizon_stability
```

Intent:

- reward getting closer to the intended geometric layout
- encourage even angular spacing
- keep the formation radius consistent around the target

## Risk-aware navigation

```text
R_risk = w_progress * goal_progress
       + w_reached * newly_reached_goals
       + w_exposure * step_risk_exposure
```

Intent:

- keep navigation objective intact
- make risk visible as a first-class optimization signal
- separate soft exposure cost from hard no-fly violations handled in shared penalties
