from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.greedy_goal import _extract_peaks
from envs import CentralizedMultiUAVEnv
from policies import build_policy, observation_to_tensor
from scripts._common import ensure_dir, load_generic_config, observation_override_from_variant, prepare_env_config


def _goal_channel(observation: dict[str, np.ndarray]) -> np.ndarray:
    task_field = observation["task_field"]
    if task_field.shape[0] == 1:
        return task_field[0]
    return task_field[1]


def _alignment_stats(observation: dict[str, np.ndarray], action: np.ndarray) -> dict[str, float]:
    positions = observation["agents"][:, :2]
    map_size = float(observation["global_info"][-1])
    goals = _extract_peaks(_goal_channel(observation), max(len(positions), 1), map_size=map_size)
    if len(goals) == 0:
        return {
            "action_abs_mean": float(np.abs(action).mean()),
            "action_norm_mean": float(np.linalg.norm(action, axis=-1).mean()),
            "nearest_goal_alignment_mean": 0.0,
            "moving_toward_goal_frac": 0.0,
            "remaining_peak_count": 0.0,
        }
    nearest = goals[np.argmin(np.linalg.norm(positions[:, None, :] - goals[None, :, :], axis=-1), axis=1)]
    target_vec = nearest - positions
    action_norm = np.linalg.norm(action, axis=-1)
    target_norm = np.linalg.norm(target_vec, axis=-1)
    denom = np.maximum(action_norm * target_norm, 1e-8)
    alignment = (action * target_vec).sum(axis=-1) / denom
    valid = target_norm > 1e-6
    if not np.any(valid):
        alignment_mean = 0.0
        moving_frac = 0.0
    else:
        alignment_mean = float(alignment[valid].mean())
        moving_frac = float((alignment[valid] > 0.25).mean())
    return {
        "action_abs_mean": float(np.abs(action).mean()),
        "action_norm_mean": float(action_norm.mean()),
        "nearest_goal_alignment_mean": alignment_mean,
        "moving_toward_goal_frac": moving_frac,
        "remaining_peak_count": float(len(goals)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy-config", required=True)
    parser.add_argument("--env-config", default="configs/env/multitask.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--obs_variant", default="multi_channel_field+task_id")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    policy_config = load_generic_config(args.policy_config)
    env_config = prepare_env_config(
        args.env_config,
        tasks=["goal_nav"],
        num_agents=4,
        scaling_mode="fixed_map",
        observation_override=observation_override_from_variant(args.obs_variant),
    )
    env = CentralizedMultiUAVEnv(env_config)
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    policy.load_state_dict(checkpoint["model_state_dict"], strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()

    output_dir = ensure_dir(args.output_dir)
    step_rows: list[dict[str, float | int]] = []
    episode_rows: list[dict[str, float | int]] = []

    for episode in range(int(args.episodes)):
        observation, _ = env.reset(seed=int(args.seed) + episode)
        done = False
        total_reward = 0.0
        step = 0
        last_info = {}
        while not done:
            with torch.no_grad():
                obs_tensor = observation_to_tensor(observation, device)
                action = policy.act_deterministic(obs_tensor).squeeze(0).detach().cpu().numpy()
            stats = _alignment_stats(observation, action)
            next_observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            step_rows.append(
                {
                    "episode": episode,
                    "step": step,
                    "reward": float(reward),
                    "goal_coverage_ratio": float(info.get("goal_coverage_ratio", 0.0)),
                    "collision_rate": float(info.get("collision_rate", 0.0)),
                    "path_length": float(info.get("path_length", 0.0)),
                    **stats,
                }
            )
            observation = next_observation
            last_info = info
            done = bool(terminated or truncated)
            step += 1
        episode_rows.append(
            {
                "episode": episode,
                "return": total_reward,
                "success": float(last_info.get("success", False)),
                "goal_coverage_ratio": float(last_info.get("goal_coverage_ratio", 0.0)),
                "collision_rate": float(last_info.get("collision_rate", 0.0)),
                "path_length": float(last_info.get("path_length", 0.0)),
                "steps": step,
            }
        )

    step_path = output_dir / "goal_nav_policy_step_diagnostics.csv"
    episode_path = output_dir / "goal_nav_policy_episode_diagnostics.csv"
    with step_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(step_rows[0].keys()))
        writer.writeheader()
        writer.writerows(step_rows)
    with episode_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)

    summary = {
        "checkpoint": str(args.checkpoint),
        "episodes": int(args.episodes),
        "success_rate": float(np.mean([row["success"] for row in episode_rows])),
        "return_mean": float(np.mean([row["return"] for row in episode_rows])),
        "goal_coverage_mean": float(np.mean([row["goal_coverage_ratio"] for row in episode_rows])),
        "collision_rate_mean": float(np.mean([row["collision_rate"] for row in episode_rows])),
        "path_length_mean": float(np.mean([row["path_length"] for row in episode_rows])),
        "nearest_goal_alignment_mean": float(np.mean([row["nearest_goal_alignment_mean"] for row in step_rows])),
        "moving_toward_goal_frac": float(np.mean([row["moving_toward_goal_frac"] for row in step_rows])),
        "action_abs_mean": float(np.mean([row["action_abs_mean"] for row in step_rows])),
    }
    summary_path = output_dir / "goal_nav_policy_diagnostics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"summary={summary_path}")
    print(f"episodes={episode_path}")
    print(f"steps={step_path}")


if __name__ == "__main__":
    main()
