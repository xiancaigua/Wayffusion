from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms import PPOTrainer
from policies import build_policy
from scripts._common import (
    baseline_reference_episodes_for_agent_count,
    baseline_reference_table,
    build_metric_logger,
    checkpoints_dir,
    format_agent_set_name,
    format_obs_variant_name,
    format_task_set_name,
    log_scalar_metrics,
    load_generic_config,
    normalize_task_names,
    observation_override_from_variant,
    print_progress_line,
    prepare_env_config,
    save_run_snapshot,
    timestamped_training_dir,
    write_metrics_csv,
)
from utils import aggregate_episode_records, apply_reference_normalization, evaluate_policy_episodes, make_env_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/policy/ppo_cnn_deepsets.yaml")
    parser.add_argument("--env-config", default="configs/env/multitask.yaml")
    parser.add_argument("--tasks", nargs="+", default=["goal_nav", "coverage"])
    parser.add_argument("--agent_counts", nargs="+", type=int, default=[4])
    parser.add_argument("--scaling_mode", default="fixed_map")
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--obs_variant", default="multi_channel_field+task_id")
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--total_updates", type=int, default=None)
    parser.add_argument("--target_episodes", type=int, default=None)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--console_log_interval", type=int, default=5)
    parser.add_argument("--record_eval_episodes", type=int, default=0)
    parser.add_argument("--record_format", choices=["gif", "mp4"], default="gif")
    parser.add_argument("--record_fps", type=int, default=8)
    parser.add_argument("--record_interval", type=int, default=1)
    args = parser.parse_args()

    task_names = normalize_task_names(args.tasks)
    agent_counts = [int(n) for n in args.agent_counts]
    train_config = load_generic_config(args.config)
    if args.total_updates is not None:
        train_config["total_updates"] = int(args.total_updates)
    if args.target_episodes is not None:
        train_config["target_episodes"] = int(args.target_episodes)
    variable_n = len(set(agent_counts)) > 1
    if variable_n and train_config["policy_class"] == "mlp":
        raise ValueError("Flatten MLP PPO is restricted to fixed-N training. Use one agent count or switch architecture.")

    build_num_agents = agent_counts[0] if train_config["policy_class"] == "mlp" else max(agent_counts)
    base_env_config = prepare_env_config(
        args.env_config,
        tasks=task_names,
        num_agents=build_num_agents,
        scaling_mode=args.scaling_mode,
        observation_override=observation_override_from_variant(args.obs_variant),
    )
    env_build = make_env_batch(base_env_config, int(train_config.get("num_envs", 1)))
    policy = build_policy(train_config, env_build.envs[0].observation_space, env_build.envs[0].action_space)
    trainer = PPOTrainer(env_build, policy, train_config)
    if args.init_checkpoint:
        trainer.load_checkpoint(args.init_checkpoint)

    output_root = "bc_ppo" if args.init_checkpoint else "ppo"
    run_name = f"{train_config['name']}_{format_task_set_name(task_names)}_N{format_agent_set_name(agent_counts)}_{format_obs_variant_name(args.obs_variant)}"
    output_dir = timestamped_training_dir(
        output_root,
        run_name,
    )
    save_run_snapshot(
        output_dir,
        train_config=train_config,
        env_config=base_env_config,
        cli_args=vars(args),
        model_state_dict=trainer.policy.state_dict(),
        extra_metadata={"task_names": task_names, "agent_counts": agent_counts, "output_root": output_root},
    )
    writer, log_record = build_metric_logger(
        output_dir,
        namespace=f"{output_root}/train",
        step_key="update",
        tensorboard_enabled=args.tensorboard,
        console_interval=args.console_log_interval,
        key_order=[
            "mean_rollout_reward",
            "policy_loss",
            "value_loss",
            "entropy",
            "episodes_completed",
            "cumulative_episodes",
            "rollout_steps_per_sec",
            "eval_reward",
            "eval_success_rate",
        ],
    )

    try:
        if not variable_n:
            history = trainer.train(
                output_dir,
                eval_env=env_build.envs[0],
                eval_episodes=args.eval_episodes,
                headless=args.headless,
                record_eval_episodes=args.record_eval_episodes,
                record_format=args.record_format,
                record_fps=args.record_fps,
                record_interval=args.record_interval,
                log_callback=log_record,
            )
        else:
            env_batches = {
                n: make_env_batch(
                    prepare_env_config(
                        args.env_config,
                        tasks=task_names,
                        num_agents=n,
                        scaling_mode=args.scaling_mode,
                        observation_override=observation_override_from_variant(args.obs_variant),
                    ),
                    int(train_config.get("num_envs", 1)),
                )
                for n in agent_counts
            }
            history = []
            total_updates = int(train_config["total_updates"])
            target_episodes = int(train_config.get("target_episodes", 0) or 0)
            eval_interval = int(train_config.get("eval_interval", total_updates))
            rng = np.random.default_rng(int(base_env_config.get("seed", 0)))
            eval_count = 0
            for update_idx in range(1, total_updates + 1):
                trainer._set_lr(update_idx, total_updates)
                chosen_n = int(rng.choice(agent_counts))
                trainer.set_env_batch(env_batches[chosen_n])
                batch, rollout_stats = trainer.collect_rollout()
                train_stats = trainer.update(batch)
                record = {"update": update_idx, "sampled_num_agents": chosen_n, **rollout_stats, **train_stats}
                if update_idx % eval_interval == 0 or update_idx == total_updates:
                    eval_count += 1
                    eval_env = env_batches[agent_counts[0]].envs[0]
                    media_dir = None
                    media_episodes = 0
                    if args.record_eval_episodes > 0 and eval_count % max(args.record_interval, 1) == 0:
                        media_dir = output_dir / "media" / f"eval_{update_idx:04d}"
                        media_episodes = int(args.record_eval_episodes)
                    record.update(
                        trainer.evaluate(
                            eval_env,
                            episodes=args.eval_episodes,
                            headless=args.headless,
                            record_dir=media_dir,
                            record_episodes=media_episodes,
                            record_format=args.record_format,
                            record_fps=args.record_fps,
                            record_prefix=f"update_{update_idx:04d}",
                        )
                    )
                    checkpoint_path = checkpoints_dir(output_dir) / f"checkpoint_{update_idx:04d}.pt"
                    torch.save({"model_state_dict": trainer.policy.state_dict(), "train_config": train_config}, checkpoint_path)
                    record["checkpoint_path"] = str(checkpoint_path)
                history.append(record)
                log_record(record)
                if target_episodes > 0 and trainer.completed_episodes >= target_episodes:
                    break
        write_metrics_csv(history, output_dir / "training_metrics.csv")

        eval_records = []
        final_update = int(history[-1].get("update", 0)) if history else 0
        for agent_count in agent_counts:
            eval_env_config = prepare_env_config(
                args.env_config,
                tasks=task_names,
                num_agents=agent_count,
                scaling_mode=args.scaling_mode,
                observation_override=observation_override_from_variant(args.obs_variant),
            )
            from envs import CentralizedMultiUAVEnv

            eval_env = CentralizedMultiUAVEnv(eval_env_config)
            final_media_dir = output_dir / "final_eval_media" / f"N{agent_count}" if args.record_eval_episodes > 0 else None
            records = evaluate_policy_episodes(
                eval_env,
                trainer.policy,
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
                eval_env_config,
                episodes=baseline_reference_episodes_for_agent_count(agent_count, max(4, args.eval_episodes // 2)),
            )
            normalized_records = [apply_reference_normalization(record, reference_table) for record in records]
            summary = aggregate_episode_records(normalized_records)
            log_scalar_metrics(writer, f"{output_root}/final_eval/N{agent_count}", final_update, summary)
            print_progress_line(
                f"{output_root}-final",
                "num_agents",
                agent_count,
                summary,
                key_order=["return_mean", "normalized_score_mean", "success_rate_mean", "collision_rate_mean"],
            )
            eval_records.append(
                {
                    "method": "bc_ppo" if args.init_checkpoint else "ppo",
                    "algorithm": "ppo",
                    "architecture": train_config["policy_class"],
                    "observation_mode": eval_env_config.get("observation_mode", "multi_channel_field"),
                    "task_set": format_task_set_name(task_names),
                    "task_name": "multitask" if len(task_names) > 1 else task_names[0],
                    "num_agents": agent_count,
                    "scaling_mode": args.scaling_mode,
                    "seed": int(eval_env_config.get("seed", 0)),
                    "normalized_score": float(summary.get("normalized_score_mean", 0.0)),
                    **summary,
                }
            )
        write_metrics_csv(eval_records, output_dir / "eval_metrics.csv")
    finally:
        if writer is not None:
            writer.close()
    print(f"ppo_output={output_dir}")


if __name__ == "__main__":
    main()
