from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common import (
    baseline_reference_episodes_for_agent_count,
    baseline_reference_table,
    format_obs_variant_name,
    format_task_set_name,
    latest_checkpoint,
    load_generic_config,
    normalize_task_names,
    observation_override_from_variant,
    prepare_env_config,
    run_dir_from_checkpoint,
    write_metrics_csv,
)
from scripts.evaluate_scaling import evaluate_baseline, load_policy_for_counts
from utils import aggregate_episode_records, apply_reference_normalization, evaluate_policy_episodes


METHOD_TO_DIR = {
    "ppo": Path("outputs/training/ppo"),
    "bc": Path("outputs/training/bc"),
    "bc_ppo": Path("outputs/training/bc_ppo"),
    "sac": Path("outputs/training/sac"),
    "td3": Path("outputs/training/td3"),
}

METHOD_TO_CONFIG = {
    "ppo": "configs/policy/ppo_cnn_deepsets.yaml",
    "bc": "configs/policy/bc_cnn_deepsets.yaml",
    "bc_ppo": "configs/policy/ppo_from_bc.yaml",
    "sac": "configs/policy/sac_cnn_deepsets.yaml",
    "td3": "configs/policy/td3_cnn_deepsets.yaml",
}


def find_checkpoint(method: str) -> Path | None:
    base_dir = METHOD_TO_DIR.get(method)
    if base_dir is None or not base_dir.exists():
        return None
    config_path = METHOD_TO_CONFIG.get(method)
    preferred_prefix = None
    if config_path is not None:
        preferred_prefix = str(load_generic_config(config_path).get("name", "")).strip()
    candidates = list(base_dir.rglob("checkpoint*.pt"))
    if preferred_prefix:
        preferred = [path for path in candidates if run_dir_from_checkpoint(path).name.startswith(preferred_prefix)]
        if preferred:
            candidates = preferred
    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["ppo", "bc", "bc_ppo"])
    parser.add_argument("--tasks", nargs="+", default=["goal_nav", "coverage"])
    parser.add_argument("--agent_counts", nargs="+", type=int, default=[4, 8, 10])
    parser.add_argument("--env-config", default="configs/env/multitask.yaml")
    parser.add_argument("--scaling_mode", default="fixed_map")
    parser.add_argument("--obs_variant", default="multi_channel_field+task_id")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output-path", default="outputs/eval/algorithm_comparison.csv")
    args = parser.parse_args()

    task_names = normalize_task_names(args.tasks)
    records = []
    for method in args.configs:
        if method in {"heuristic", "random"}:
            for agent_count in args.agent_counts:
                env_config = prepare_env_config(
                    args.env_config,
                    tasks=task_names,
                    num_agents=int(agent_count),
                    scaling_mode=args.scaling_mode,
                    observation_override=observation_override_from_variant(args.obs_variant),
                )
                summary = evaluate_baseline(method, env_config, args.episodes)
                records.append(
                    {
                        "method": method,
                        "algorithm": method,
                        "architecture": "baseline",
                        "observation_mode": env_config.get("observation_mode", "multi_channel_field"),
                        "task_set": format_task_set_name(task_names),
                        "task_name": "multitask" if len(task_names) > 1 else task_names[0],
                        "num_agents": int(agent_count),
                        "scaling_mode": args.scaling_mode,
                        "seed": int(env_config.get("seed", 0)),
                        "normalized_score": float(summary.get("normalized_score_mean", 0.0)),
                        **summary,
                    }
                )
            continue

        checkpoint = find_checkpoint(method)
        if checkpoint is None:
            continue
        policy, policy_config, device = load_policy_for_counts(
            checkpoint_path=checkpoint,
            policy_config_path=METHOD_TO_CONFIG[method],
            env_config_path=args.env_config,
            task_names=task_names,
            agent_counts=[int(n) for n in args.agent_counts],
            scaling_mode=args.scaling_mode,
            observation_override=observation_override_from_variant(args.obs_variant),
        )
        for agent_count in args.agent_counts:
            env_config = prepare_env_config(
                args.env_config,
                tasks=task_names,
                num_agents=int(agent_count),
                scaling_mode=args.scaling_mode,
                observation_override=observation_override_from_variant(args.obs_variant),
            )
            from envs import CentralizedMultiUAVEnv

            env = CentralizedMultiUAVEnv(env_config)
            episode_records = evaluate_policy_episodes(env, policy, args.episodes, device)
            reference_table = baseline_reference_table(
                env_config,
                episodes=baseline_reference_episodes_for_agent_count(int(agent_count), max(4, args.episodes)),
            )
            normalized_records = [apply_reference_normalization(record, reference_table) for record in episode_records]
            summary = aggregate_episode_records(normalized_records)
            records.append(
                {
                    "method": method,
                    "algorithm": policy_config.get("algorithm", method),
                    "architecture": policy_config["policy_class"],
                    "observation_mode": env_config.get("observation_mode", "multi_channel_field"),
                    "task_set": format_task_set_name(task_names),
                    "task_name": "multitask" if len(task_names) > 1 else task_names[0],
                    "num_agents": int(agent_count),
                    "scaling_mode": args.scaling_mode,
                    "seed": int(env_config.get("seed", 0)),
                    "normalized_score": float(summary.get("normalized_score_mean", 0.0)),
                    **summary,
                }
            )

    write_metrics_csv(records, args.output_path)
    print(f"algorithm_comparison={args.output_path}")


if __name__ == "__main__":
    main()
