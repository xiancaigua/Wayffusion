from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines import make_baseline
from policies import build_policy
from scripts._common import (
    baseline_reference_episodes_for_agent_count,
    collect_baseline_episode_records,
    ensure_dir,
    format_agent_set_name,
    format_obs_variant_name,
    format_task_set_name,
    load_generic_config,
    normalize_task_names,
    observation_override_from_variant,
    prepare_env_config,
    write_metrics_csv,
)
from utils import evaluate_policy_per_task


def evaluate_baseline(method: str, env_config: dict, episodes: int) -> dict:
    from utils import aggregate_episode_records, apply_reference_normalization

    records = collect_baseline_episode_records(env_config, method, episodes)
    from scripts._common import baseline_reference_table

    reference_table = baseline_reference_table(
        env_config,
        episodes=baseline_reference_episodes_for_agent_count(int(env_config["num_agents"]), episodes),
    )
    normalized_records = [apply_reference_normalization(record, reference_table) for record in records]
    return aggregate_episode_records(normalized_records)


def evaluate_baseline_per_task(method: str, base_env_config: dict, task_names: list[str], episodes: int) -> tuple[list[dict], dict[str, dict], dict]:
    from scripts._common import baseline_reference_table
    from utils import aggregate_episode_records, apply_reference_normalization, make_fixed_task_eval_config

    all_records: list[dict] = []
    task_summaries: dict[str, dict] = {}
    for task_name in task_names:
        eval_config = make_fixed_task_eval_config(base_env_config, task_name)
        records = collect_baseline_episode_records(eval_config, method, episodes)
        reference_table = baseline_reference_table(
            eval_config,
            episodes=baseline_reference_episodes_for_agent_count(int(eval_config["num_agents"]), episodes),
        )
        normalized_records = [apply_reference_normalization(record, reference_table) for record in records]
        for record in normalized_records:
            record["task_name"] = task_name
            record["eval_group"] = task_name
        summary = aggregate_episode_records(normalized_records)
        summary["task_name"] = task_name
        summary["eval_group"] = task_name
        task_summaries[task_name] = summary
        all_records.extend(normalized_records)
    overall_summary = aggregate_episode_records(all_records)
    overall_summary["task_name"] = "overall"
    overall_summary["eval_group"] = "overall"
    return all_records, task_summaries, overall_summary


def load_policy_for_counts(
    checkpoint_path: str | Path,
    policy_config_path: str | Path,
    env_config_path: str | Path,
    task_names: list[str],
    agent_counts: list[int],
    scaling_mode: str,
    observation_override: dict | None = None,
):
    policy_config = load_generic_config(policy_config_path)
    build_num_agents = max(agent_counts) if policy_config["policy_class"] != "mlp" else agent_counts[0]
    build_env_config = prepare_env_config(
        env_config_path,
        tasks=task_names,
        num_agents=build_num_agents,
        scaling_mode=scaling_mode,
        observation_override=observation_override,
    )
    from envs import CentralizedMultiUAVEnv

    build_env = CentralizedMultiUAVEnv(build_env_config)
    policy = build_policy(policy_config, build_env.observation_space, build_env.action_space)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    policy.load_state_dict(checkpoint["model_state_dict"], strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    return policy, policy_config, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--policy-config", default="configs/policy/ppo_cnn_deepsets.yaml")
    parser.add_argument("--method", default="policy")
    parser.add_argument("--tasks", nargs="+", default=["goal_nav", "coverage"])
    parser.add_argument("--agent_counts", nargs="+", type=int, required=True)
    parser.add_argument("--env-config", default="configs/env/multitask.yaml")
    parser.add_argument("--scaling_mode", default="fixed_map")
    parser.add_argument("--obs_variant", default="multi_channel_field+task_id")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output-path", default="outputs/eval/scaling_fixed_N.csv")
    parser.add_argument("--protocol", choices=["fixed_N", "variable_N"], default="fixed_N")
    parser.add_argument("--train_agent_counts", nargs="+", type=int, default=None)
    args = parser.parse_args()

    task_names = normalize_task_names(args.tasks)
    agent_counts = [int(n) for n in args.agent_counts]
    train_agent_counts = [int(n) for n in args.train_agent_counts] if args.train_agent_counts else list(agent_counts)
    records = []
    policy = None
    policy_config = None
    device = None
    if args.checkpoint:
        policy, policy_config, device = load_policy_for_counts(
            checkpoint_path=args.checkpoint,
            policy_config_path=args.policy_config,
            env_config_path=args.env_config,
            task_names=task_names,
            agent_counts=agent_counts,
            scaling_mode=args.scaling_mode,
            observation_override=observation_override_from_variant(args.obs_variant),
        )

    overall_rows = []
    for agent_count in agent_counts:
        env_config = prepare_env_config(
            args.env_config,
            tasks=task_names,
            num_agents=agent_count,
            scaling_mode=args.scaling_mode,
            observation_override=observation_override_from_variant(args.obs_variant),
        )
        if args.checkpoint:
            method_name = args.method
            algorithm = policy_config.get("algorithm", args.method)
            architecture = policy_config["policy_class"]
            _, task_summaries, overall_summary = evaluate_policy_per_task(
                env_config,
                policy,
                task_names,
                args.episodes,
                device,
                normalize_with_reference=True,
                reference_episodes=baseline_reference_episodes_for_agent_count(agent_count, max(4, args.episodes // 2)),
            )
        else:
            method_name = args.method
            algorithm = args.method
            architecture = "heuristic" if args.method == "heuristic" else "baseline"
            _, task_summaries, overall_summary = evaluate_baseline_per_task(args.method, env_config, task_names, args.episodes)
        for task_name, task_summary in task_summaries.items():
            records.append(
                {
                    **task_summary,
                    "method": method_name,
                    "algorithm": algorithm,
                    "architecture": architecture,
                    "observation_mode": env_config.get("observation_mode", "multi_channel_field"),
                    "obs_variant": args.obs_variant,
                    "task_set": format_task_set_name(task_names),
                    "task_name": task_name,
                    "eval_group": "per_task",
                    "num_agents": agent_count,
                    "scaling_mode": args.scaling_mode,
                    "protocol": args.protocol,
                    "train_agent_set": format_agent_set_name(train_agent_counts),
                    "seed": int(env_config.get("seed", 0)),
                    "normalized_score": float(task_summary.get("normalized_score_mean", 0.0)),
                }
            )
        overall_row = (
            {
                **overall_summary,
                "method": method_name,
                "algorithm": algorithm,
                "architecture": architecture,
                "observation_mode": env_config.get("observation_mode", "multi_channel_field"),
                "obs_variant": args.obs_variant,
                "task_set": format_task_set_name(task_names),
                "task_name": "overall",
                "eval_group": "overall",
                "num_agents": agent_count,
                "scaling_mode": args.scaling_mode,
                "protocol": args.protocol,
                "train_agent_set": format_agent_set_name(train_agent_counts),
                "seed": int(env_config.get("seed", 0)),
                "normalized_score": float(overall_summary.get("normalized_score_mean", 0.0)),
            }
        )
        records.append(overall_row)
        overall_rows.append(overall_row)

    if overall_rows:
        normalized_values = [float(record["normalized_score"]) for record in overall_rows]
        average_normalized_score = float(np.mean(normalized_values))
        worst_task_score = float(np.min(normalized_values))
        for record in records:
            record["average_normalized_score"] = average_normalized_score
            record["worst_task_score"] = worst_task_score

    output_path = ensure_dir(Path(args.output_path).parent) / Path(args.output_path).name
    write_metrics_csv(records, output_path)
    print(f"scaling_eval={output_path}")


if __name__ == "__main__":
    main()
