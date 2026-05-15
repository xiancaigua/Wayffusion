from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines import HeuristicPolicy
from scripts._common import (
    ensure_dir,
    format_agent_set_name,
    format_obs_variant_name,
    format_task_set_name,
    normalize_task_names,
    observation_override_from_variant,
    prepare_env_config,
)
from utils import save_expert_dataset


def infer_output_path(task_names: list[str], agent_counts: list[int], obs_variant: str) -> Path:
    output_dir = ensure_dir("outputs/datasets")
    obs_tag = format_obs_variant_name(obs_variant)
    if len(task_names) == 1 and len(agent_counts) == 1:
        return output_dir / f"expert_{task_names[0]}_N{agent_counts[0]}_{obs_tag}.npz"
    return output_dir / f"expert_{format_task_set_name(task_names)}_N{format_agent_set_name(agent_counts)}_{obs_tag}.npz"


def pad_agents(array: np.ndarray, max_agents: int, fill_value: float = 0.0) -> np.ndarray:
    padded = np.full((max_agents, array.shape[-1]), fill_value, dtype=np.float32)
    padded[: array.shape[0]] = array.astype(np.float32)
    return padded


def pad_action(array: np.ndarray, max_agents: int) -> np.ndarray:
    padded = np.zeros((max_agents, 2), dtype=np.float32)
    padded[: array.shape[0]] = array.astype(np.float32)
    return padded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--agent_counts", nargs="+", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--config", default="configs/env/multitask.yaml")
    parser.add_argument("--scaling_mode", default="fixed_map")
    parser.add_argument("--obs_variant", default="multi_channel_field+task_id")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    task_names = normalize_task_names(args.tasks)
    agent_counts = [int(n) for n in args.agent_counts]
    output_path = Path(args.output) if args.output else infer_output_path(task_names, agent_counts, args.obs_variant)
    max_agents = max(agent_counts)

    task_fields = []
    agents = []
    task_ids = []
    global_infos = []
    agent_masks = []
    actions = []
    rewards = []
    dones = []
    task_name_ids = []
    num_agents_list = []
    intrinsic_scores = []
    success_rates = []

    for agent_count in agent_counts:
        env_config = prepare_env_config(
            args.config,
            tasks=task_names,
            num_agents=agent_count,
            scaling_mode=args.scaling_mode,
            observation_override=observation_override_from_variant(args.obs_variant),
        )
        policy = HeuristicPolicy(env_config)
        env = None
        for episode_idx in range(args.episodes):
            if env is None:
                from envs import CentralizedMultiUAVEnv

                env = CentralizedMultiUAVEnv(env_config)
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
                task_name_ids.append(str(info["task_name"]))
                num_agents_list.append(int(agent_count))
                intrinsic_scores.append(float(info.get("intrinsic_score", 0.0)))
                success_rates.append(float(info.get("success", False)))
                obs = next_obs
                done = terminated or truncated

    payload = {
        "task_field": np.stack(task_fields, axis=0).astype(np.float32),
        "agents": np.stack(agents, axis=0).astype(np.float32),
        "task_id": np.stack(task_ids, axis=0).astype(np.float32),
        "global_info": np.stack(global_infos, axis=0).astype(np.float32),
        "agent_mask": np.stack(agent_masks, axis=0).astype(np.float32),
        "action": np.stack(actions, axis=0).astype(np.float32),
        "reward": np.asarray(rewards, dtype=np.float32),
        "done": np.asarray(dones, dtype=np.float32),
        "task_name": np.asarray(task_name_ids, dtype=object),
        "num_agents": np.asarray(num_agents_list, dtype=np.int32),
        "intrinsic_score": np.asarray(intrinsic_scores, dtype=np.float32),
        "success": np.asarray(success_rates, dtype=np.float32),
    }
    save_expert_dataset(output_path, payload)
    print(f"saved_dataset={output_path}")


if __name__ == "__main__":
    main()
