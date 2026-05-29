from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import CentralizedMultiUAVEnv
from policies import build_policy, observation_to_tensor
from scripts._common import load_generic_config, normalize_task_names, observation_override_from_variant, prepare_env_config
from utils import load_expert_dataset, save_expert_dataset


def _atomic_save_dataset(path: Path, payload: dict[str, np.ndarray]) -> None:
    tmp_path = path.with_name(path.stem + ".tmp" + path.suffix)
    save_expert_dataset(tmp_path, payload)
    os.replace(tmp_path, path)


def _atomic_write_text(path: Path, content: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def _pad_agents(array: np.ndarray, max_agents: int) -> np.ndarray:
    padded = np.zeros((max_agents, array.shape[-1]), dtype=np.float32)
    padded[: array.shape[0]] = array.astype(np.float32)
    return padded


def _pad_action(array: np.ndarray, max_agents: int) -> np.ndarray:
    padded = np.zeros((max_agents, 2), dtype=np.float32)
    padded[: array.shape[0]] = array.astype(np.float32)
    return padded


def _empty_payload() -> dict[str, list]:
    return {
        "task_field": [],
        "agents": [],
        "task_id": [],
        "global_info": [],
        "agent_mask": [],
        "action": [],
        "reward": [],
        "done": [],
        "task_name": [],
        "num_agents": [],
        "intrinsic_score": [],
        "success": [],
    }


def _extend_npz(payload: dict[str, list], dataset_path: str | Path) -> int:
    data = load_expert_dataset(dataset_path)
    count = int(data["action"].shape[0])
    for key in payload:
        payload[key].extend(list(data[key]))
    return count


def _append_sample(payload: dict[str, list], observation: dict, action: np.ndarray, reward: float, done: bool, info: dict, max_agents: int) -> None:
    count = observation["agents"].shape[0]
    mask = np.zeros((max_agents,), dtype=np.float32)
    mask[:count] = 1.0
    payload["task_field"].append(observation["task_field"].astype(np.float32))
    payload["agents"].append(_pad_agents(observation["agents"], max_agents))
    payload["task_id"].append(observation["task_id"].astype(np.float32))
    payload["global_info"].append(observation["global_info"].astype(np.float32))
    payload["agent_mask"].append(mask)
    payload["action"].append(_pad_action(action, max_agents))
    payload["reward"].append(float(reward))
    payload["done"].append(float(done))
    payload["task_name"].append(str(info["task_name"]))
    payload["num_agents"].append(int(count))
    payload["intrinsic_score"].append(float(info.get("intrinsic_score", 0.0)))
    payload["success"].append(float(info.get("success", False)))


def _stack_payload(payload: dict[str, list]) -> dict[str, np.ndarray]:
    return {
        "task_field": np.stack(payload["task_field"], axis=0).astype(np.float32),
        "agents": np.stack(payload["agents"], axis=0).astype(np.float32),
        "task_id": np.stack(payload["task_id"], axis=0).astype(np.float32),
        "global_info": np.stack(payload["global_info"], axis=0).astype(np.float32),
        "agent_mask": np.stack(payload["agent_mask"], axis=0).astype(np.float32),
        "action": np.stack(payload["action"], axis=0).astype(np.float32),
        "reward": np.asarray(payload["reward"], dtype=np.float32),
        "done": np.asarray(payload["done"], dtype=np.float32),
        "task_name": np.asarray(payload["task_name"], dtype=object),
        "num_agents": np.asarray(payload["num_agents"], dtype=np.int32),
        "intrinsic_score": np.asarray(payload["intrinsic_score"], dtype=np.float32),
        "success": np.asarray(payload["success"], dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy-config", required=True)
    parser.add_argument("--base_dataset", default=None)
    parser.add_argument("--tasks", nargs="+", default=["coverage"])
    parser.add_argument("--agent_counts", nargs="+", type=int, default=[4])
    parser.add_argument("--success_episodes", type=int, default=20)
    parser.add_argument("--max_attempts", type=int, default=240)
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--env-config", default="configs/env/multitask.yaml")
    parser.add_argument("--scaling_mode", default="fixed_map")
    parser.add_argument("--obs_variant", default="multi_channel_field+task_id")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=5200)
    parser.add_argument("--action_noise_std", type=float, default=0.0)
    args = parser.parse_args()

    task_names = normalize_task_names(args.tasks)
    agent_counts = [int(n) for n in args.agent_counts]
    max_agents = max(agent_counts)
    policy_config = load_generic_config(args.policy_config)
    env_config = prepare_env_config(
        args.env_config,
        tasks=task_names,
        num_agents=max_agents,
        scaling_mode=args.scaling_mode,
        observation_override=observation_override_from_variant(args.obs_variant),
    )
    env = CentralizedMultiUAVEnv(env_config)
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    policy.load_state_dict(checkpoint["model_state_dict"], strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()

    payload = _empty_payload()
    base_samples = _extend_npz(payload, args.base_dataset) if args.base_dataset else 0
    successful_episodes = 0
    attempts = 0
    summaries = []
    sample_stride = max(int(args.sample_stride), 1)

    while attempts < int(args.max_attempts) and successful_episodes < int(args.success_episodes):
        observation, _ = env.reset(seed=int(args.seed) + attempts)
        done = False
        total_reward = 0.0
        step = 0
        episode_payload = _empty_payload()
        info = {}
        while not done:
            with torch.no_grad():
                action = (
                    policy.act_deterministic(observation_to_tensor(observation, device))
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                )
            if float(args.action_noise_std) > 0.0:
                action = np.clip(
                    action + np.random.normal(scale=float(args.action_noise_std), size=action.shape).astype(np.float32),
                    -1.0,
                    1.0,
                )
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            if step % sample_stride == 0 or done:
                _append_sample(episode_payload, observation, action, reward, done, info, max_agents)
            observation = next_observation
            step += 1

        success = bool(info.get("success", False))
        summaries.append(
            {
                "attempt": attempts,
                "seed": int(args.seed) + attempts,
                "success": float(success),
                "return": total_reward,
                "coverage_ratio": float(info.get("coverage_ratio", 0.0)),
                "collision_rate": float(info.get("collision_rate", 0.0)),
                "path_length": float(info.get("path_length", 0.0)),
                "steps": step,
                "samples": len(episode_payload["action"]),
            }
        )
        attempts += 1
        if not success:
            continue
        if len(payload["action"]) + len(episode_payload["action"]) > int(args.max_samples):
            break
        successful_episodes += 1
        for key, values in episode_payload.items():
            payload[key].extend(values)

    if not payload["action"]:
        raise RuntimeError("No successful policy episodes were collected.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save_dataset(output_path, _stack_payload(payload))
    summary = {
        "output": str(output_path),
        "checkpoint": str(args.checkpoint),
        "base_dataset": str(args.base_dataset) if args.base_dataset else None,
        "action_noise_std": float(args.action_noise_std),
        "base_samples": base_samples,
        "attempts": attempts,
        "successful_episodes": successful_episodes,
        "samples": len(payload["action"]),
        "sample_stride": sample_stride,
        "max_samples": int(args.max_samples),
        "success_rate_over_attempts": float(successful_episodes / max(attempts, 1)),
        "mean_success_return": float(np.mean([row["return"] for row in summaries if row["success"] > 0.5])),
        "mean_success_path_length": float(np.mean([row["path_length"] for row in summaries if row["success"] > 0.5])),
        "mean_success_collision_rate": float(np.mean([row["collision_rate"] for row in summaries if row["success"] > 0.5])),
    }
    _atomic_write_text(output_path.with_suffix(".summary.json"), json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _atomic_write_text(output_path.with_suffix(".episodes.json"), json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    print(f"saved_dataset={output_path}")
    print(f"summary={output_path.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
