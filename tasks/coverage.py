from __future__ import annotations

import numpy as np

from fields.field_utils import gaussian_map, normalize_map, sample_points
from tasks.base_task import BaseTask, TaskStepResult


class CoverageTask(BaseTask):
    name = "coverage"
    task_id = 1

    def _success_ratio(self, task_state, env_state) -> float:
        required_visits = max(int(task_state.get("required_visits", 1)), 1)
        success_ratio = float(self.config["coverage"]["success_ratio"]) + 0.05 * max(required_visits - 1, 0)
        return min(success_ratio, 0.999)

    def reset(self, rng: np.random.Generator, env_state: dict) -> dict:
        multi_peak = bool(self.config["coverage"]["multi_peak_probability"])
        task_count_scale = env_state.get("task_count_scale", 1.0)
        scaling_mode = str(env_state.get("scaling_mode", "fixed_map"))
        density_peak_scale_exponent = float(self.config["coverage"].get("density_peak_scale_exponent", 0.8))
        peak_scale = task_count_scale if scaling_mode == "fixed_map" else task_count_scale ** density_peak_scale_exponent
        peak_count = (5 if multi_peak else 2) + max(0, int(round(peak_scale)) - 1)
        required_visits = 1
        if scaling_mode == "fixed_map":
            required_visits = max(1, int(np.ceil(np.sqrt(task_count_scale))))
        else:
            required_visits = max(1, int(np.ceil(task_count_scale ** 0.25)))
        centers = sample_points(rng, peak_count, margin=0.08, map_size=env_state["map_size"])
        amplitudes = rng.uniform(0.5, 1.0, size=(peak_count,)).astype(np.float32)
        target_probability = normalize_map(
            gaussian_map(
                centers,
                env_state["grid_size"],
                sigma=0.09 * env_state["spatial_scale"],
                amplitudes=amplitudes,
                map_size=env_state["map_size"],
            )
        )
        demand_quantile = float(self.config["coverage"].get("demand_quantile", 0.55))
        demand_quantile = float(np.clip(demand_quantile, 0.0, 0.99))
        coverage_demand = (target_probability > np.quantile(target_probability, demand_quantile)).astype(np.float32)
        return {
            "target_probability": target_probability,
            "coverage_demand": coverage_demand,
            "required_visits": required_visits,
            "route_hint_routes": None,
            "route_hint_indices": None,
            "last_coverage_ratio": 0.0,
            "last_detection_score": 0.0,
            "success_bonus_paid": False,
            "paid_milestones": set(),
            "total_repeated": 0.0,
            "total_detected": 0.0,
        }

    def build_field(self, task_state, env_state) -> dict:
        field = {
            "target_probability": task_state["target_probability"],
            "desired_occupancy": np.clip(
                task_state["coverage_demand"] * min(float(task_state["required_visits"]) / 4.0, 1.0),
                0.0,
                1.0,
            ),
        }
        if bool(self.config["coverage"].get("route_hint_enabled", False)):
            field["formation_template"] = self._route_hint_map(task_state, env_state)
        return field

    def _route_hint_map(self, task_state: dict, env_state: dict) -> np.ndarray:
        """Expose persistent per-agent sweep targets through an existing field channel.

        The route state is environment/task state, not mutable policy state, so
        PPO can still recompute policy log-probs from recorded observations.
        """

        positions = np.asarray(env_state["positions"], dtype=np.float32)
        num_agents = int(len(positions))
        if num_agents <= 0:
            return np.zeros_like(task_state["coverage_demand"], dtype=np.float32)
        routes = task_state.get("route_hint_routes")
        route_indices = task_state.get("route_hint_indices")
        if routes is None or route_indices is None or len(routes) != num_agents:
            routes, route_indices = self._build_route_hints(task_state, env_state)
            task_state["route_hint_routes"] = routes
            task_state["route_hint_indices"] = route_indices

        targets = self._advance_route_hints(task_state, env_state)
        task_state["route_hint_targets"] = targets.astype(np.float32)
        sigma = float(self.config["coverage"].get("route_hint_sigma", 0.035)) * float(env_state.get("spatial_scale", 1.0))
        hint = gaussian_map(
            targets,
            int(env_state["grid_size"]),
            sigma=max(sigma, 1e-4),
            amplitudes=np.ones((num_agents,), dtype=np.float32),
            map_size=float(env_state["map_size"]),
        )
        return np.clip(hint, 0.0, 1.0).astype(np.float32)

    def _build_route_hints(self, task_state: dict, env_state: dict) -> tuple[list[np.ndarray], np.ndarray]:
        positions = np.asarray(env_state["positions"], dtype=np.float32)
        demand = np.asarray(task_state["coverage_demand"], dtype=bool)
        obstacle = np.asarray(env_state["obstacle_map"], dtype=np.float32) >= 0.4
        valid = demand & ~obstacle
        if not np.any(valid):
            valid = demand
        ys, xs = np.where(valid)
        num_agents = int(len(positions))
        if len(xs) == 0:
            return [positions[idx : idx + 1].copy() for idx in range(num_agents)], np.zeros((num_agents,), dtype=np.int32)

        stride = max(int(self.config["coverage"].get("route_hint_stride", 3)), 1)
        keep = ((xs + ys) % stride) == 0
        if np.any(keep):
            xs = xs[keep]
            ys = ys[keep]
        map_size = float(env_state["map_size"])
        height, width = task_state["coverage_demand"].shape
        points = np.stack(
            [
                xs / max(width - 1, 1) * map_size,
                ys / max(height - 1, 1) * map_size,
            ],
            axis=-1,
        ).astype(np.float32)
        scores = task_state["target_probability"][ys, xs].astype(np.float32)

        x_order = np.argsort(positions[:, 0])
        quantiles = np.quantile(points[:, 0], np.linspace(0.0, 1.0, num_agents + 1))
        routes: list[np.ndarray] = [np.zeros((0, 2), dtype=np.float32) for _ in range(num_agents)]
        y_bin_count = max(3, min(9, int(np.sqrt(max(len(points), 1))) + 1))
        y_bins = np.linspace(0.0, map_size, y_bin_count)
        value_quantile = float(self.config["coverage"].get("route_hint_value_quantile", 0.15))
        for stripe_idx, agent_idx in enumerate(x_order):
            lo = quantiles[stripe_idx] - 1e-6
            hi = quantiles[stripe_idx + 1] + 1e-6
            in_stripe = (points[:, 0] >= lo) & (points[:, 0] <= hi)
            stripe_points = points[in_stripe]
            stripe_scores = scores[in_stripe]
            if len(stripe_points) == 0:
                continue
            cutoff = np.quantile(stripe_scores, value_quantile) if len(stripe_scores) > 8 else float(stripe_scores.min())
            stripe_points = stripe_points[stripe_scores >= cutoff]
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
            start_idx = int(np.argmin(np.linalg.norm(route - positions[int(agent_idx)], axis=1)))
            routes[int(agent_idx)] = np.concatenate([route[start_idx:], route[:start_idx]], axis=0).astype(np.float32)

        for agent_idx in range(num_agents):
            if len(routes[agent_idx]) == 0:
                nearest_idx = int(np.argmin(np.linalg.norm(points - positions[agent_idx], axis=1)))
                routes[agent_idx] = points[nearest_idx : nearest_idx + 1]
        return routes, np.zeros((num_agents,), dtype=np.int32)

    @staticmethod
    def _grid_value(grid: np.ndarray, point: np.ndarray, map_size: float) -> float:
        height, width = grid.shape
        x_idx = int(np.clip(round(float(point[0]) / max(map_size, 1e-6) * (width - 1)), 0, width - 1))
        y_idx = int(np.clip(round(float(point[1]) / max(map_size, 1e-6) * (height - 1)), 0, height - 1))
        return float(grid[y_idx, x_idx])

    def _advance_route_hints(self, task_state: dict, env_state: dict) -> np.ndarray:
        positions = np.asarray(env_state["positions"], dtype=np.float32)
        routes = task_state["route_hint_routes"]
        route_indices = task_state["route_hint_indices"]
        visited = np.asarray(env_state["visited_map"], dtype=np.float32)
        remaining = task_state["coverage_demand"] * np.clip(1.0 - visited, 0.0, 1.0)
        map_size = float(env_state["map_size"])
        coverage_radius = float(env_state.get("coverage_radius", self.config.get("coverage_radius", 0.05)))
        targets = np.zeros_like(positions, dtype=np.float32)
        for agent_idx, route in enumerate(routes):
            if len(route) == 0:
                targets[agent_idx] = positions[agent_idx]
                continue
            for _ in range(len(route)):
                target = route[int(route_indices[agent_idx]) % len(route)]
                distance = float(np.linalg.norm(target - positions[agent_idx]))
                remaining_at_target = self._grid_value(remaining, target, map_size)
                if distance > 0.65 * coverage_radius and remaining_at_target > 0.05:
                    break
                route_indices[agent_idx] = (int(route_indices[agent_idx]) + 1) % len(route)
            targets[agent_idx] = route[int(route_indices[agent_idx]) % len(route)]
        return targets.astype(np.float32)

    def compute_reward(self, task_state, prev_env_state, env_state, transition_info) -> TaskStepResult:
        required_visits = max(int(task_state.get("required_visits", 1)), 1)
        prev_fulfillment = np.clip(prev_env_state["visit_count_map"] / required_visits, 0.0, 1.0)
        current_fulfillment = np.clip(env_state["visit_count_map"] / required_visits, 0.0, 1.0)
        new_mask = np.clip(current_fulfillment - prev_fulfillment, 0.0, 1.0)
        repeated_mask = (prev_env_state["visit_count_map"] >= required_visits).astype(np.float32) * env_state["step_coverage_mask"]
        repeated_demand_mask = repeated_mask * task_state["coverage_demand"]

        new_coverage = float((new_mask * task_state["coverage_demand"]).sum() / np.maximum(task_state["coverage_demand"].sum(), 1.0))
        detection_gain = float(
            (new_mask * task_state["target_probability"]).sum()
            / np.maximum(task_state["target_probability"].sum(), 1e-6)
        )
        repeated_ratio = float(repeated_mask.sum() / np.maximum(env_state["step_coverage_mask"].sum(), 1.0))
        repeated_demand_ratio = float(repeated_demand_mask.sum() / np.maximum(task_state["coverage_demand"].sum(), 1.0))

        task_state["total_repeated"] += repeated_mask.sum()
        task_state["total_detected"] += detection_gain

        metrics = self.get_metrics(task_state, env_state)
        previous_coverage_ratio = float(task_state.get("last_coverage_ratio", 0.0))
        previous_detection_score = float(task_state.get("last_detection_score", 0.0))
        coverage_delta = max(float(metrics["coverage_ratio"]) - previous_coverage_ratio, 0.0)
        detection_delta = max(float(metrics["accumulated_detection_probability"]) - previous_detection_score, 0.0)
        task_state["last_coverage_ratio"] = float(metrics["coverage_ratio"])
        task_state["last_detection_score"] = float(metrics["accumulated_detection_probability"])

        weights = self.config["reward_weights"]["coverage"]
        success_bonus = 0.0
        if bool(metrics["success"]) and not bool(task_state.get("success_bonus_paid", False)):
            success_bonus = float(weights.get("success_bonus", 0.0))
            task_state["success_bonus_paid"] = True

        coverage_reward = float(weights["new_coverage"] * max(new_coverage, coverage_delta))
        detection_reward = float(weights["high_probability"] * max(detection_gain, detection_delta))
        coverage_level_reward = float(weights.get("coverage_level", 0.0) * float(metrics["coverage_ratio"]))
        milestone_reward = self._milestone_reward(task_state, float(metrics["coverage_ratio"]), weights)
        success_ratio = self._success_ratio(task_state, env_state)
        shortfall = max(success_ratio - float(metrics["coverage_ratio"]), 0.0)
        shortfall_penalty = float(weights.get("coverage_shortfall", 0.0) * shortfall)
        failure_penalty = 0.0
        if not bool(metrics["success"]) and env_state["step_count"] >= int(env_state["max_steps"]):
            failure_penalty = float(weights.get("failure_penalty", 0.0))
        repeated_penalty = float(weights["repeated_coverage"] * repeated_ratio)
        repeated_demand_penalty = float(weights.get("repeated_demand_coverage", 0.0) * repeated_demand_ratio)
        terminal_repeated_penalty = 0.0
        terminal_revisit_excess_penalty = 0.0
        if env_state["step_count"] >= int(env_state["max_steps"]):
            terminal_repeated_penalty = float(weights.get("terminal_repeated_coverage", 0.0) * float(metrics["repeated_coverage_ratio"]))
            terminal_revisit_excess_penalty = float(
                weights.get("terminal_revisit_excess", 0.0) * float(metrics["demand_revisit_excess"])
            )
        reward = (
            coverage_reward
            + detection_reward
            + coverage_level_reward
            + milestone_reward
            + shortfall_penalty
            + failure_penalty
            + repeated_penalty
            + repeated_demand_penalty
            + terminal_repeated_penalty
            + terminal_revisit_excess_penalty
            + success_bonus
        )
        return TaskStepResult(
            reward=reward,
            success=bool(metrics["success"]),
            metrics=metrics,
            components={
                "coverage_reward": coverage_reward,
                "detection_reward": detection_reward,
                "coverage_level_reward": coverage_level_reward,
                "coverage_milestone_reward": milestone_reward,
                "coverage_shortfall_penalty": shortfall_penalty,
                "coverage_failure_penalty": failure_penalty,
                "repeated_coverage_penalty": repeated_penalty,
                "repeated_demand_coverage_penalty": repeated_demand_penalty,
                "terminal_repeated_coverage_penalty": terminal_repeated_penalty,
                "terminal_revisit_excess_penalty": terminal_revisit_excess_penalty,
                "coverage_success_bonus": success_bonus,
            },
        )

    def _milestone_reward(self, task_state: dict, coverage_ratio: float, weights: dict) -> float:
        thresholds = weights.get("milestone_thresholds", [])
        bonuses = weights.get("milestone_bonuses", [])
        if not thresholds or not bonuses:
            return 0.0
        paid = task_state.setdefault("paid_milestones", set())
        reward = 0.0
        for index, threshold in enumerate(thresholds):
            threshold_value = float(threshold)
            if coverage_ratio < threshold_value or threshold_value in paid:
                continue
            bonus_idx = min(index, len(bonuses) - 1)
            reward += float(bonuses[bonus_idx])
            paid.add(threshold_value)
        return float(reward)

    def get_metrics(self, task_state, env_state) -> dict:
        demand = np.maximum(task_state["coverage_demand"].sum(), 1.0)
        required_visits = max(int(task_state.get("required_visits", 1)), 1)
        fulfillment_map = np.clip(env_state["visit_count_map"] / required_visits, 0.0, 1.0)
        coverage_ratio = float((fulfillment_map * task_state["coverage_demand"]).sum() / demand)
        detection_score = float(
            (fulfillment_map * task_state["target_probability"]).sum()
            / np.maximum(task_state["target_probability"].sum(), 1e-6)
        )
        visit_counts = env_state["visit_count_map"]
        repeated_ratio = float((visit_counts > required_visits).sum() / np.maximum((visit_counts > 0).sum(), 1.0))
        excess_demand_visits = np.maximum(visit_counts - required_visits, 0.0) * task_state["coverage_demand"]
        demand_revisit_excess = float(excess_demand_visits.sum() / demand)
        positions = env_state["positions"]
        centroid = positions.mean(axis=0, keepdims=True)
        dispersion = float(np.linalg.norm(positions - centroid, axis=-1).mean())
        min_steps = max(1, int(np.ceil(2.0 * np.sqrt(max(env_state.get("task_count_scale", 1.0), 1.0)))))
        success_ratio = self._success_ratio(task_state, env_state)
        success = coverage_ratio >= success_ratio and env_state["step_count"] >= min_steps
        return {
            "success": float(success),
            "coverage_ratio": coverage_ratio,
            "accumulated_detection_probability": detection_score,
            "repeated_coverage_ratio": repeated_ratio,
            "demand_revisit_excess": demand_revisit_excess,
            "time_discounted_detection_score": detection_score / (1.0 + 0.02 * env_state["step_count"]),
            "spatial_dispersion": dispersion,
        }
