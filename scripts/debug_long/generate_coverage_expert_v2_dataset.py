from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import CentralizedMultiUAVEnv
from scripts._common import ensure_dir, observation_override_from_variant, prepare_env_config
from scripts.generate_expert_dataset import pad_action, pad_agents
from utils import save_expert_dataset


def _atomic_save_dataset(path: Path, payload: dict[str, np.ndarray]) -> None:
    tmp_path = path.with_name(path.stem + ".tmp" + path.suffix)
    save_expert_dataset(tmp_path, payload)
    os.replace(tmp_path, path)


class CoverageExpertV2:
    """Coverage expert that explicitly spreads agents across high-demand cells."""

    def __init__(self, config: dict):
        self.max_waypoint_step = float(config["max_waypoint_step"])

    def act(self, observation: dict) -> np.ndarray:
        positions = observation["agents"][:, :2]
        task_field = observation["task_field"]
        map_size = float(observation["global_info"][-1])
        target_probability = task_field[2]
        desired_occupancy = task_field[3]
        risk = task_field[4]
        visited = task_field[5]

        utility = 1.8 * desired_occupancy + 0.6 * target_probability - 0.4 * visited - 0.2 * risk
        flat_utility = utility.reshape(-1)
        topk = min(256, flat_utility.size)
        candidate_idx = np.argpartition(flat_utility, -topk)[-topk:]
        ys, xs = np.unravel_index(candidate_idx, utility.shape)
        candidate_points = np.stack(
            [
                xs / max(utility.shape[1] - 1, 1) * map_size,
                ys / max(utility.shape[0] - 1, 1) * map_size,
            ],
            axis=-1,
        ).astype(np.float32)
        candidate_scores = utility[ys, xs]

        selected_targets: list[np.ndarray] = []
        remaining = list(range(len(candidate_points)))
        if remaining:
            first_idx = int(np.argmax(candidate_scores))
            selected_targets.append(candidate_points[first_idx])
            remaining.remove(first_idx)

        while len(selected_targets) < len(positions) and remaining:
            best_idx = None
            best_score = -1e18
            for idx in remaining:
                point = candidate_points[idx]
                spread_bonus = 0.0
                if selected_targets:
                    spread_bonus = min(float(np.linalg.norm(point - target)) for target in selected_targets)
                score = float(candidate_scores[idx]) + 0.2 * spread_bonus
                if score > best_score:
                    best_score = score
                    best_idx = idx
            selected_targets.append(candidate_points[best_idx])
            remaining.remove(best_idx)

        if not selected_targets:
            return np.zeros((len(positions), 2), dtype=np.float32)

        targets = np.asarray(selected_targets, dtype=np.float32)
        distances = np.linalg.norm(positions[:, None, :] - targets[None, :, :], axis=-1)
        rows, cols = linear_sum_assignment(distances)
        ordered_targets = np.zeros_like(positions, dtype=np.float32)
        ordered_targets[rows] = targets[cols]
        for agent_idx in range(len(positions)):
            if not np.any(ordered_targets[agent_idx]):
                ordered_targets[agent_idx] = targets[int(np.argmin(distances[agent_idx]))]
        return np.clip((ordered_targets - positions) / max(self.max_waypoint_step, 1e-6), -1.0, 1.0).astype(np.float32)


class CoverageExpertV3:
    """Local frontier coverage expert with short-range collision avoidance."""

    def __init__(self, config: dict, topk: int = 512):
        self.max_waypoint_step = float(config["max_waypoint_step"])
        self.collision_radius = float(config["collision_radius"])
        self.topk = int(topk)

    def act(self, observation: dict) -> np.ndarray:
        positions = observation["agents"][:, :2]
        task_field = observation["task_field"]
        map_size = float(observation["global_info"][-1])
        obstacle = task_field[0]
        target_probability = task_field[2]
        desired_occupancy = task_field[3]
        risk = task_field[4]
        visited = task_field[5]
        agent_density = task_field[6]

        remaining = np.clip(1.0 - visited, 0.0, 1.0)
        demand = (desired_occupancy > 0.0).astype(np.float32)
        utility = (3.0 * demand + 0.9 * target_probability) * remaining
        utility = utility - 0.9 * risk - 2.2 * obstacle - 0.35 * agent_density

        flat_utility = utility.reshape(-1)
        topk = min(self.topk, flat_utility.size)
        candidate_idx = np.argpartition(flat_utility, -topk)[-topk:]
        ys, xs = np.unravel_index(candidate_idx, utility.shape)
        candidate_points = np.stack(
            [
                xs / max(utility.shape[1] - 1, 1) * map_size,
                ys / max(utility.shape[0] - 1, 1) * map_size,
            ],
            axis=-1,
        ).astype(np.float32)
        candidate_scores = flat_utility[candidate_idx].astype(np.float32)

        selected_targets: list[np.ndarray] = []
        target_pairs: list[tuple[int, np.ndarray]] = []
        used = np.zeros((len(candidate_points),), dtype=bool)
        centroid = positions.mean(axis=0)
        agent_order = np.argsort(np.linalg.norm(positions - centroid, axis=1))[::-1]
        for agent_idx in agent_order:
            position = positions[agent_idx]
            distances = np.linalg.norm(candidate_points - position, axis=1)
            local_bonus = np.exp(-distances / 0.22)
            if selected_targets:
                selected = np.asarray(selected_targets, dtype=np.float32)
                spread_bonus = np.linalg.norm(candidate_points[:, None, :] - selected[None, :, :], axis=-1).min(axis=1)
            else:
                spread_bonus = np.full((len(candidate_points),), 0.2, dtype=np.float32)
            scores = 4.5 * candidate_scores + 1.3 * local_bonus + 1.1 * spread_bonus - 0.4 * distances
            scores[used] = -1e9
            best_idx = int(np.argmax(scores))
            used[best_idx] = True
            selected_targets.append(candidate_points[best_idx])
            target_pairs.append((int(agent_idx), candidate_points[best_idx]))

        ordered_targets = np.zeros_like(positions, dtype=np.float32)
        for agent_idx, target in target_pairs:
            ordered_targets[agent_idx] = target
        action = (ordered_targets - positions) / max(self.max_waypoint_step, 1e-6)

        rel = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(rel, axis=-1, keepdims=True)
        avoidance_radius = 3.0 * self.collision_radius
        mask = (distances > 1e-6) & (distances < avoidance_radius)
        repulsion_scale = (avoidance_radius - np.minimum(distances, avoidance_radius)) / max(avoidance_radius, 1e-6)
        repulsion = np.where(mask, rel / np.maximum(distances, 1e-6) * repulsion_scale, 0.0).sum(axis=1)
        action = action + 0.8 * repulsion
        return np.clip(action, -1.0, 1.0).astype(np.float32)


class CoverageExpertV4:
    """Stateful sector coverage expert with persistent per-agent targets."""

    def __init__(self, config: dict, topk: int = 768):
        self.max_waypoint_step = float(config["max_waypoint_step"])
        self.collision_radius = float(config["collision_radius"])
        self.topk = int(topk)
        self.targets: np.ndarray | None = None
        self.target_age: np.ndarray | None = None
        self.last_progress = -1.0

    def _reset_if_needed(self, progress: float, num_agents: int) -> None:
        if self.targets is None or self.targets.shape[0] != num_agents or progress <= self.last_progress:
            self.targets = np.full((num_agents, 2), np.nan, dtype=np.float32)
            self.target_age = np.zeros((num_agents,), dtype=np.int32)
        self.last_progress = progress

    @staticmethod
    def _grid_value(grid: np.ndarray, point: np.ndarray, map_size: float) -> float:
        h, w = grid.shape
        x = int(np.clip(round(float(point[0]) / max(map_size, 1e-6) * (w - 1)), 0, w - 1))
        y = int(np.clip(round(float(point[1]) / max(map_size, 1e-6) * (h - 1)), 0, h - 1))
        return float(grid[y, x])

    def _select_target(
        self,
        agent_idx: int,
        sector_idx: int,
        positions: np.ndarray,
        candidate_points: np.ndarray,
        candidate_scores: np.ndarray,
        used: np.ndarray,
        map_size: float,
    ) -> np.ndarray:
        num_agents = max(len(positions), 1)
        center = np.full((2,), 0.5 * map_size, dtype=np.float32)
        rel = candidate_points - center[None, :]
        angles = (np.arctan2(rel[:, 1], rel[:, 0]) + 2.0 * np.pi) % (2.0 * np.pi)
        target_angle = (float(sector_idx) + 0.5) / float(num_agents) * 2.0 * np.pi
        angle_delta = np.abs((angles - target_angle + np.pi) % (2.0 * np.pi) - np.pi)
        sector_bonus = np.exp(-angle_delta / max(np.pi / num_agents, 1e-6))

        distances = np.linalg.norm(candidate_points - positions[agent_idx], axis=1)
        if np.any(np.isfinite(self.targets).all(axis=1)):
            finite_targets = self.targets[np.isfinite(self.targets).all(axis=1)]
            spread = np.linalg.norm(candidate_points[:, None, :] - finite_targets[None, :, :], axis=-1).min(axis=1)
        else:
            spread = np.full((len(candidate_points),), 0.2 * map_size, dtype=np.float32)
        scores = 6.0 * candidate_scores + 1.5 * sector_bonus + 0.9 * spread - 0.35 * distances
        scores[used] = -1e9
        best_idx = int(np.argmax(scores))
        used[best_idx] = True
        return candidate_points[best_idx]

    def act(self, observation: dict) -> np.ndarray:
        positions = observation["agents"][:, :2]
        task_field = observation["task_field"]
        progress = float(observation["global_info"][0])
        map_size = float(observation["global_info"][-1])
        self._reset_if_needed(progress, len(positions))

        obstacle = task_field[0]
        target_probability = task_field[2]
        desired_occupancy = task_field[3]
        risk = task_field[4]
        visited = task_field[5]
        agent_density = task_field[6]

        demand = (desired_occupancy > 0.0).astype(np.float32)
        remaining = np.clip(1.0 - visited, 0.0, 1.0)
        utility = (4.0 * demand + 1.2 * target_probability) * remaining
        utility = utility - 2.5 * obstacle - 0.5 * risk - 0.25 * agent_density

        flat_utility = utility.reshape(-1)
        topk = min(self.topk, flat_utility.size)
        candidate_idx = np.argpartition(flat_utility, -topk)[-topk:]
        ys, xs = np.unravel_index(candidate_idx, utility.shape)
        candidate_points = np.stack(
            [
                xs / max(utility.shape[1] - 1, 1) * map_size,
                ys / max(utility.shape[0] - 1, 1) * map_size,
            ],
            axis=-1,
        ).astype(np.float32)
        candidate_scores = flat_utility[candidate_idx].astype(np.float32)
        used = np.zeros((len(candidate_points),), dtype=bool)

        center = np.full((2,), 0.5 * map_size, dtype=np.float32)
        agent_angles = (np.arctan2(positions[:, 1] - center[1], positions[:, 0] - center[0]) + 2.0 * np.pi) % (2.0 * np.pi)
        agent_order = np.argsort(agent_angles)
        sector_for_agent = np.zeros((len(positions),), dtype=np.int32)
        for sector_idx, agent_idx in enumerate(agent_order):
            sector_for_agent[int(agent_idx)] = int(sector_idx)

        assert self.targets is not None
        assert self.target_age is not None
        for agent_idx in range(len(positions)):
            target = self.targets[agent_idx]
            keep_target = bool(np.isfinite(target).all())
            if keep_target:
                distance_to_target = float(np.linalg.norm(target - positions[agent_idx]))
                remaining_at_target = self._grid_value(remaining * demand, target, map_size)
                keep_target = distance_to_target > 0.035 and remaining_at_target > 0.15 and self.target_age[agent_idx] < 35
            if not keep_target:
                self.targets[agent_idx] = self._select_target(
                    agent_idx,
                    int(sector_for_agent[agent_idx]),
                    positions,
                    candidate_points,
                    candidate_scores,
                    used,
                    map_size,
                )
                self.target_age[agent_idx] = 0
            else:
                self.target_age[agent_idx] += 1

        action = (self.targets - positions) / max(self.max_waypoint_step, 1e-6)
        rel = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(rel, axis=-1, keepdims=True)
        avoidance_radius = 3.5 * self.collision_radius
        mask = (distances > 1e-6) & (distances < avoidance_radius)
        repulsion_scale = (avoidance_radius - np.minimum(distances, avoidance_radius)) / max(avoidance_radius, 1e-6)
        repulsion = np.where(mask, rel / np.maximum(distances, 1e-6) * repulsion_scale, 0.0).sum(axis=1)
        action = action + 0.9 * repulsion
        return np.clip(action, -1.0, 1.0).astype(np.float32)


class CoverageExpertV5:
    """Stateful lawnmower-route coverage expert for diagnostic teacher data.

    The expert builds one persistent route per agent by splitting remaining
    demand cells into x-stripes and ordering each stripe in alternating y
    direction. This is intentionally non-learned and diagnostic: it tests
    whether route persistence solves the repeated-coverage failure mode better
    than per-step local frontier scoring.
    """

    def __init__(self, config: dict):
        self.max_waypoint_step = float(config["max_waypoint_step"])
        self.coverage_radius = float(config["coverage_radius"])
        self.collision_radius = float(config["collision_radius"])
        self.routes: list[np.ndarray] | None = None
        self.route_indices: np.ndarray | None = None
        self.last_progress = -1.0

    def _reset_if_needed(self, progress: float, num_agents: int) -> None:
        if self.routes is None or self.route_indices is None or len(self.routes) != num_agents or progress <= self.last_progress:
            self.routes = None
            self.route_indices = None
        self.last_progress = progress

    @staticmethod
    def _grid_value(grid: np.ndarray, point: np.ndarray, map_size: float) -> float:
        h, w = grid.shape
        x = int(np.clip(round(float(point[0]) / max(map_size, 1e-6) * (w - 1)), 0, w - 1))
        y = int(np.clip(round(float(point[1]) / max(map_size, 1e-6) * (h - 1)), 0, h - 1))
        return float(grid[y, x])

    def _build_routes(self, observation: dict) -> None:
        positions = observation["agents"][:, :2]
        task_field = observation["task_field"]
        map_size = float(observation["global_info"][-1])
        desired_occupancy = task_field[3]
        target_probability = task_field[2]
        obstacle = task_field[0]
        visited = task_field[5]
        demand = (desired_occupancy > 0.0) & (obstacle < 0.4)
        remaining = demand & (visited < 0.95)
        if not np.any(remaining):
            remaining = demand

        ys, xs = np.where(remaining)
        if len(xs) == 0:
            self.routes = [positions[idx : idx + 1].copy() for idx in range(len(positions))]
            self.route_indices = np.zeros((len(positions),), dtype=np.int32)
            return

        stride = 3
        keep = ((xs + ys) % stride) == 0
        xs = xs[keep]
        ys = ys[keep]
        if len(xs) == 0:
            ys, xs = np.where(remaining)
        points = np.stack(
            [
                xs / max(desired_occupancy.shape[1] - 1, 1) * map_size,
                ys / max(desired_occupancy.shape[0] - 1, 1) * map_size,
            ],
            axis=-1,
        ).astype(np.float32)
        scores = target_probability[ys, xs].astype(np.float32)

        num_agents = len(positions)
        x_order = np.argsort(positions[:, 0])
        quantiles = np.quantile(points[:, 0], np.linspace(0.0, 1.0, num_agents + 1))
        routes: list[np.ndarray] = [np.zeros((0, 2), dtype=np.float32) for _ in range(num_agents)]
        for stripe_idx, agent_idx in enumerate(x_order):
            lo = quantiles[stripe_idx] - 1e-6
            hi = quantiles[stripe_idx + 1] + 1e-6
            in_stripe = (points[:, 0] >= lo) & (points[:, 0] <= hi)
            stripe_points = points[in_stripe]
            stripe_scores = scores[in_stripe]
            if len(stripe_points) == 0:
                continue
            # Keep higher-value cells first, then create an alternating sweep
            # order to reduce long jumps within the assigned stripe.
            value_cutoff = np.quantile(stripe_scores, 0.15) if len(stripe_scores) > 8 else float(stripe_scores.min())
            stripe_points = stripe_points[stripe_scores >= value_cutoff]
            y_bins = np.linspace(0.0, map_size, max(3, min(9, int(np.sqrt(len(stripe_points))) + 1)))
            ordered_chunks = []
            for bin_idx in range(len(y_bins) - 1):
                in_bin = (stripe_points[:, 1] >= y_bins[bin_idx]) & (stripe_points[:, 1] <= y_bins[bin_idx + 1])
                chunk = stripe_points[in_bin]
                if len(chunk) == 0:
                    continue
                order = np.argsort(chunk[:, 0])
                if bin_idx % 2:
                    order = order[::-1]
                ordered_chunks.append(chunk[order])
            route = np.concatenate(ordered_chunks, axis=0) if ordered_chunks else stripe_points
            start_idx = int(np.argmin(np.linalg.norm(route - positions[agent_idx], axis=1)))
            routes[int(agent_idx)] = np.concatenate([route[start_idx:], route[:start_idx]], axis=0).astype(np.float32)

        for agent_idx in range(num_agents):
            if len(routes[agent_idx]) == 0:
                nearest_idx = int(np.argmin(np.linalg.norm(points - positions[agent_idx], axis=1)))
                routes[agent_idx] = points[nearest_idx : nearest_idx + 1]
        self.routes = routes
        self.route_indices = np.zeros((num_agents,), dtype=np.int32)

    def act(self, observation: dict) -> np.ndarray:
        positions = observation["agents"][:, :2]
        progress = float(observation["global_info"][0])
        map_size = float(observation["global_info"][-1])
        task_field = observation["task_field"]
        desired_occupancy = task_field[3]
        visited = task_field[5]
        remaining = (desired_occupancy > 0.0).astype(np.float32) * np.clip(1.0 - visited, 0.0, 1.0)
        self._reset_if_needed(progress, len(positions))
        if self.routes is None or self.route_indices is None:
            self._build_routes(observation)
        assert self.routes is not None
        assert self.route_indices is not None

        targets = np.zeros_like(positions, dtype=np.float32)
        for agent_idx, route in enumerate(self.routes):
            if len(route) == 0:
                targets[agent_idx] = positions[agent_idx]
                continue
            for _ in range(len(route)):
                target = route[int(self.route_indices[agent_idx]) % len(route)]
                distance = float(np.linalg.norm(target - positions[agent_idx]))
                remaining_at_target = self._grid_value(remaining, target, map_size)
                if distance > 0.65 * self.coverage_radius and remaining_at_target > 0.05:
                    break
                self.route_indices[agent_idx] = (self.route_indices[agent_idx] + 1) % len(route)
            targets[agent_idx] = route[int(self.route_indices[agent_idx]) % len(route)]

        action = (targets - positions) / max(self.max_waypoint_step, 1e-6)
        rel = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(rel, axis=-1, keepdims=True)
        avoidance_radius = 3.0 * self.collision_radius
        mask = (distances > 1e-6) & (distances < avoidance_radius)
        repulsion_scale = (avoidance_radius - np.minimum(distances, avoidance_radius)) / max(avoidance_radius, 1e-6)
        repulsion = np.where(mask, rel / np.maximum(distances, 1e-6) * repulsion_scale, 0.0).sum(axis=1)
        action = action + 0.6 * repulsion
        return np.clip(action, -1.0, 1.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--agent_count", type=int, default=4)
    parser.add_argument("--env-config", default="configs/env/multitask.yaml")
    parser.add_argument("--obs_variant", default="multi_channel_field+task_id")
    parser.add_argument("--expert", choices=["v2", "v3", "v4", "v5"], default="v2")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    env_config = prepare_env_config(
        args.env_config,
        tasks=["coverage"],
        num_agents=int(args.agent_count),
        scaling_mode="fixed_map",
        observation_override=observation_override_from_variant(args.obs_variant),
    )
    env = CentralizedMultiUAVEnv(env_config)
    if args.expert == "v5":
        policy = CoverageExpertV5(env_config)
    elif args.expert == "v4":
        policy = CoverageExpertV4(env_config)
    elif args.expert == "v3":
        policy = CoverageExpertV3(env_config)
    else:
        policy = CoverageExpertV2(env_config)
    max_agents = int(args.agent_count)

    task_fields = []
    agents = []
    task_ids = []
    global_infos = []
    agent_masks = []
    actions = []
    rewards = []
    dones = []
    task_names = []
    num_agents = []
    intrinsic_scores = []
    success_rates = []

    for episode_idx in range(int(args.episodes)):
        obs, _ = env.reset(seed=int(env_config.get("seed", 0)) + episode_idx)
        done = False
        while not done:
            action = policy.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            count = obs["agents"].shape[0]
            task_fields.append(obs["task_field"].astype(np.float32))
            agents.append(pad_agents(obs["agents"], max_agents))
            task_ids.append(obs["task_id"].astype(np.float32))
            global_infos.append(obs["global_info"].astype(np.float32))
            mask = np.zeros((max_agents,), dtype=np.float32)
            mask[:count] = 1.0
            agent_masks.append(mask)
            actions.append(pad_action(action, max_agents))
            rewards.append(float(reward))
            dones.append(float(terminated or truncated))
            task_names.append(str(info["task_name"]))
            num_agents.append(int(args.agent_count))
            intrinsic_scores.append(float(info.get("intrinsic_score", 0.0)))
            success_rates.append(float(info.get("success", False)))
            obs = next_obs
            done = bool(terminated or truncated)

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    payload = {
        "task_field": np.stack(task_fields, axis=0).astype(np.float32),
        "agents": np.stack(agents, axis=0).astype(np.float32),
        "task_id": np.stack(task_ids, axis=0).astype(np.float32),
        "global_info": np.stack(global_infos, axis=0).astype(np.float32),
        "agent_mask": np.stack(agent_masks, axis=0).astype(np.float32),
        "action": np.stack(actions, axis=0).astype(np.float32),
        "reward": np.asarray(rewards, dtype=np.float32),
        "done": np.asarray(dones, dtype=np.float32),
        "task_name": np.asarray(task_names, dtype=object),
        "num_agents": np.asarray(num_agents, dtype=np.int32),
        "intrinsic_score": np.asarray(intrinsic_scores, dtype=np.float32),
        "success": np.asarray(success_rates, dtype=np.float32),
    }
    _atomic_save_dataset(output_path, payload)
    print(f"saved_dataset={output_path}")


if __name__ == "__main__":
    main()
