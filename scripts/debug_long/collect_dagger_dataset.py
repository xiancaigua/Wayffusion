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

from baselines import HeuristicPolicy
from envs import CentralizedMultiUAVEnv
from policies import build_policy, observation_to_tensor
from scripts.debug_long.generate_coverage_expert_v2_dataset import CoverageExpertV2
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


def _append_sample(payload: dict[str, list], observation: dict, teacher_action: np.ndarray, reward: float, done: bool, info: dict, max_agents: int) -> None:
    count = observation["agents"].shape[0]
    mask = np.zeros((max_agents,), dtype=np.float32)
    mask[:count] = 1.0
    payload["task_field"].append(observation["task_field"].astype(np.float32))
    payload["agents"].append(_pad_agents(observation["agents"], max_agents))
    payload["task_id"].append(observation["task_id"].astype(np.float32))
    payload["global_info"].append(observation["global_info"].astype(np.float32))
    payload["agent_mask"].append(mask)
    payload["action"].append(_pad_action(teacher_action, max_agents))
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
    parser.add_argument("--tasks", nargs="+", default=["goal_nav"])
    parser.add_argument("--agent_counts", nargs="+", type=int, default=[4])
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--sample_stride", type=int, default=2)
    parser.add_argument("--max_new_samples", type=int, default=12000)
    parser.add_argument("--env-config", default="configs/env/multitask.yaml")
    parser.add_argument("--scaling_mode", default="fixed_map")
    parser.add_argument("--obs_variant", default="multi_channel_field+task_id")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2500)
    parser.add_argument("--teacher", choices=["heuristic", "coverage_expert_v2"], default="heuristic")
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
    build_env = CentralizedMultiUAVEnv(env_config)
    learner = build_policy(policy_config, build_env.observation_space, build_env.action_space)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    learner.load_state_dict(checkpoint["model_state_dict"], strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learner.to(device)
    learner.eval()

    payload = _empty_payload()
    base_samples = _extend_npz(payload, args.base_dataset) if args.base_dataset else 0
    new_samples = 0
    summaries = []
    sample_stride = max(int(args.sample_stride), 1)

    env = CentralizedMultiUAVEnv(env_config)
    if args.teacher == "coverage_expert_v2":
        teacher = CoverageExpertV2(env_config)
    else:
        teacher = HeuristicPolicy(env_config)
    for episode in range(int(args.episodes)):
        if new_samples >= int(args.max_new_samples):
            break
        observation, _ = env.reset(seed=int(args.seed) + episode)
        done = False
        step = 0
        total_reward = 0.0
        info = {}
        while not done:
            teacher_action = teacher.act(observation)
            with torch.no_grad():
                learner_action = (
                    learner.act_deterministic(observation_to_tensor(observation, device))
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                )
            next_observation, reward, terminated, truncated, info = env.step(learner_action)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            if step % sample_stride == 0 or done:
                _append_sample(payload, observation, teacher_action, reward, done, info, max_agents)
                new_samples += 1
                if new_samples >= int(args.max_new_samples):
                    break
            observation = next_observation
            step += 1
        summaries.append(
            {
                "episode": episode,
                "return": total_reward,
                "success": float(info.get("success", False)),
                "goal_coverage_ratio": float(info.get("goal_coverage_ratio", 0.0)),
                "collision_rate": float(info.get("collision_rate", 0.0)),
                "path_length": float(info.get("path_length", 0.0)),
                "steps": step,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save_dataset(output_path, _stack_payload(payload))
    summary = {
        "output": str(output_path),
        "checkpoint": str(args.checkpoint),
        "base_dataset": str(args.base_dataset) if args.base_dataset else None,
        "teacher": args.teacher,
        "base_samples": base_samples,
        "new_samples": new_samples,
        "total_samples": len(payload["action"]),
        "learner_rollout_success_rate": float(np.mean([row["success"] for row in summaries])) if summaries else 0.0,
        "learner_rollout_goal_coverage_mean": float(np.mean([row["goal_coverage_ratio"] for row in summaries])) if summaries else 0.0,
        "learner_rollout_collision_rate_mean": float(np.mean([row["collision_rate"] for row in summaries])) if summaries else 0.0,
    }
    _atomic_write_text(output_path.with_suffix(".summary.json"), json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _atomic_write_text(output_path.with_suffix(".episodes.json"), json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    print(f"saved_dataset={output_path}")
    print(f"summary={output_path.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
