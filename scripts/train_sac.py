from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms import SACTrainer
from policies import build_policy
from scripts._common import (
    baseline_reference_episodes_for_agent_count,
    baseline_reference_table,
    format_agent_set_name,
    format_obs_variant_name,
    format_task_set_name,
    load_generic_config,
    normalize_task_names,
    observation_override_from_variant,
    prepare_env_config,
    save_run_snapshot,
    timestamped_training_dir,
    write_metrics_csv,
)
from utils import aggregate_episode_records, apply_reference_normalization, evaluate_policy_episodes, make_env_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/policy/sac_cnn_deepsets.yaml")
    parser.add_argument("--env-config", default="configs/env/multitask.yaml")
    parser.add_argument("--tasks", nargs="+", default=["goal_nav", "coverage"])
    parser.add_argument("--agent_counts", nargs="+", type=int, default=[4])
    parser.add_argument("--scaling_mode", default="fixed_map")
    parser.add_argument("--obs_variant", default="multi_channel_field+task_id")
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record_eval_episodes", type=int, default=0)
    parser.add_argument("--record_format", choices=["gif", "mp4"], default="gif")
    parser.add_argument("--record_fps", type=int, default=8)
    parser.add_argument("--record_interval", type=int, default=1)
    args = parser.parse_args()

    task_names = normalize_task_names(args.tasks)
    agent_count = int(args.agent_counts[0])
    config = load_generic_config(args.config)
    env_config = prepare_env_config(
        args.env_config,
        tasks=task_names,
        num_agents=agent_count,
        scaling_mode=args.scaling_mode,
        observation_override=observation_override_from_variant(args.obs_variant),
    )
    env_batch = make_env_batch(env_config, int(config.get("num_envs", 1)))
    policy = build_policy(config, env_batch.envs[0].observation_space, env_batch.envs[0].action_space)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        policy.load_state_dict(checkpoint["model_state_dict"], strict=False)
    trainer = SACTrainer(env_batch, policy, config)
    output_root = "bc_sac" if args.init_checkpoint else "sac"
    run_name = f"{config['name']}_{format_task_set_name(task_names)}_N{format_agent_set_name([agent_count])}_{format_obs_variant_name(args.obs_variant)}"
    output_dir = timestamped_training_dir(output_root, run_name)
    save_run_snapshot(
        output_dir,
        train_config=config,
        env_config=env_config,
        cli_args=vars(args),
        model_state_dict=trainer.actor.state_dict(),
        extra_metadata={"task_names": task_names, "agent_counts": [agent_count], "output_root": output_root},
    )
    metrics = trainer.train(
        output_dir,
        eval_env=env_batch.envs[0],
        eval_episodes=args.eval_episodes,
        headless=args.headless,
        record_eval_episodes=args.record_eval_episodes,
        record_format=args.record_format,
        record_fps=args.record_fps,
        record_interval=args.record_interval,
    )
    write_metrics_csv(metrics, output_dir / "training_metrics.csv")

    from envs import CentralizedMultiUAVEnv

    eval_env = CentralizedMultiUAVEnv(env_config)
    final_media_dir = output_dir / "final_eval_media" if args.record_eval_episodes > 0 else None
    eval_records = evaluate_policy_episodes(
        eval_env,
        trainer.actor,
        args.eval_episodes,
        trainer.device,
        headless=args.headless,
        record_dir=final_media_dir,
        record_episodes=min(int(args.record_eval_episodes), int(args.eval_episodes)),
        record_format=args.record_format,
        record_fps=args.record_fps,
        record_prefix=f"final_eval_N{agent_count}",
    )
    reference_table = baseline_reference_table(
        env_config,
        episodes=baseline_reference_episodes_for_agent_count(agent_count, max(4, args.eval_episodes // 2)),
    )
    normalized_records = [apply_reference_normalization(record, reference_table) for record in eval_records]
    summary = aggregate_episode_records(normalized_records)
    write_metrics_csv(
        [
            {
                "method": "bc_sac" if args.init_checkpoint else "sac",
                "algorithm": "sac",
                "architecture": config["policy_class"],
                "observation_mode": env_config.get("observation_mode", "multi_channel_field"),
                "task_set": format_task_set_name(task_names),
                "task_name": "multitask" if len(task_names) > 1 else task_names[0],
                "num_agents": agent_count,
                "scaling_mode": args.scaling_mode,
                "seed": int(env_config.get("seed", 0)),
                "normalized_score": float(summary.get("normalized_score_mean", 0.0)),
                **summary,
            }
        ],
        output_dir / "eval_metrics.csv",
    )
    print(f"sac_output={output_dir}")


if __name__ == "__main__":
    main()
