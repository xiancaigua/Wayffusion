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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--agent_count", type=int, default=4)
    parser.add_argument("--env-config", default="configs/env/multitask.yaml")
    parser.add_argument("--obs_variant", default="multi_channel_field+task_id")
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
