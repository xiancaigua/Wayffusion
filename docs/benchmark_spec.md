# Benchmark Spec

## Goal

Build a centralized multi-UAV multi-task waypoint benchmark for first-stage AI conference experimentation. The benchmark is designed for algorithm validation, not real flight.

The policy sees:

- a global multi-channel task field,
- all agent states,
- a task identifier,
- optional global summary features.

The policy outputs a joint waypoint action for the full swarm in one shot.

## Why centralized single-agent RL

This benchmark is intentionally not MARL.

- The action is a single joint tensor `[N, 2]`, not one action head per agent.
- The observation is global and shared, not decentralized.
- The research question is whether a single centralized controller can use a unified task field to solve multiple heterogeneous spatial tasks with a shared waypoint interface.

This isolates the intended contribution:

1. task representation unification,
2. action interface unification,
3. centralized joint decision making,
4. cross-task generalization.

## Core design decisions

- World: continuous 2D square `[0, 1] x [0, 1]`
- Visualization grid: default `64 x 64`
- UAV state: `[x, y, vx, vy, battery, role_id]`
- Action: relative waypoint delta in `[-1, 1]`, scaled by `max_waypoint_step`
- Low-level execution: proportional controller with clipped velocity
- Shared observation structure for all tasks
- Fixed task field channel order with zero-filled inactive channels

## Supported tasks

- `goal_nav`: multi-goal navigation with implicit assignment
- `coverage`: search and coverage over a target probability field
- `formation`: encirclement / structure maintenance around a target
- `risk_nav`: navigation with risk and no-fly constraints

## Observation ablations

- `multi_channel`: full 9-channel task field
- `single_channel`: weighted fusion into one channel
- `task_id_only`: task field zeroed out, keeping `agents + task_id + global_info`

These ablations let the benchmark evaluate whether the spatial task field itself carries useful inductive bias.
