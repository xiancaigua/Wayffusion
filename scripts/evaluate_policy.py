from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policies import build_policy
from scripts._common import (
    baseline_reference_episodes_for_agent_count,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy-config", default="configs/policy/ppo_cnn_deepsets.yaml")
    parser.add_argument("--env-config", default="configs/env/multitask.yaml")
    parser.add_argument("--tasks", nargs="+", default=["goal_nav", "coverage"])
    parser.add_argument("--agent_counts", nargs="+", type=int, default=[4])
    parser.add_argument("--scaling_mode", default="fixed_map")
    parser.add_argument("--obs_variant", default="multi_channel_field+task_id")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/eval/policy")
    args = parser.parse_args()

    task_names = normalize_task_names(args.tasks)
    agent_counts = [int(n) for n in args.agent_counts]
    policy_config = load_generic_config(args.policy_config)
    build_num_agents = max(agent_counts) if policy_config["policy_class"] != "mlp" else agent_counts[0]
    env_config = prepare_env_config(
        args.env_config,
        tasks=task_names,
        num_agents=build_num_agents,
        scaling_mode=args.scaling_mode,
        observation_override=observation_override_from_variant(args.obs_variant),
    )
    from envs import CentralizedMultiUAVEnv

    build_env = CentralizedMultiUAVEnv(env_config)
    policy = build_policy(policy_config, build_env.observation_space, build_env.action_space)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    policy.load_state_dict(checkpoint["model_state_dict"], strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    output_dir = ensure_dir(args.output_dir)

    records = []
    for agent_count in agent_counts:
        eval_env_config = prepare_env_config(
            args.env_config,
            tasks=task_names,
            num_agents=agent_count,
            scaling_mode=args.scaling_mode,
            observation_override=observation_override_from_variant(args.obs_variant),
        )
        _, task_summaries, overall_summary = evaluate_policy_per_task(
            eval_env_config,
            policy,
            task_names,
            args.episodes,
            device,
            normalize_with_reference=True,
            reference_episodes=baseline_reference_episodes_for_agent_count(agent_count, max(4, args.episodes // 2)),
        )
        for task_name, task_summary in task_summaries.items():
            records.append(
                {
                    **task_summary,
                    "method": Path(args.checkpoint).stem,
                    "algorithm": policy_config.get("algorithm", "policy"),
                    "architecture": policy_config["policy_class"],
                    "observation_mode": eval_env_config.get("observation_mode", "multi_channel_field"),
                    "obs_variant": args.obs_variant,
                    "task_set": format_task_set_name(task_names),
                    "task_name": task_name,
                    "eval_group": "per_task",
                    "num_agents": agent_count,
                    "scaling_mode": args.scaling_mode,
                    "seed": int(eval_env_config.get("seed", 0)),
                    "normalized_score": float(task_summary.get("normalized_score_mean", 0.0)),
                }
            )
        records.append(
            {
                **overall_summary,
                "method": Path(args.checkpoint).stem,
                "algorithm": policy_config.get("algorithm", "policy"),
                "architecture": policy_config["policy_class"],
                "observation_mode": eval_env_config.get("observation_mode", "multi_channel_field"),
                "obs_variant": args.obs_variant,
                "task_set": format_task_set_name(task_names),
                "task_name": "overall",
                "eval_group": "overall",
                "num_agents": agent_count,
                "scaling_mode": args.scaling_mode,
                "seed": int(eval_env_config.get("seed", 0)),
                "normalized_score": float(overall_summary.get("normalized_score_mean", 0.0)),
            }
        )
    output_path = output_dir / f"{format_task_set_name(task_names)}_N{format_agent_set_name(agent_counts)}_{format_obs_variant_name(args.obs_variant)}.csv"
    write_metrics_csv(records, output_path)
    print(f"policy_eval={output_path}")


if __name__ == "__main__":
    main()
