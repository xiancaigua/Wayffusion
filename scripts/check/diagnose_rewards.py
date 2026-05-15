from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines import make_baseline
from scripts._common import ensure_dir, normalize_task_names, prepare_env_config, write_metrics_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--agent_counts", nargs="+", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--env-config", default="configs/env/multitask.yaml")
    parser.add_argument("--scaling_mode", default="fixed_map")
    args = parser.parse_args()

    task_names = normalize_task_names(args.tasks)
    agent_counts = [int(n) for n in args.agent_counts]
    stats = []
    plot_dir = ensure_dir("outputs/smoke/diagnostics/reward_components")
    from envs import CentralizedMultiUAVEnv

    for agent_count in agent_counts:
        for task_name in task_names:
            env_config = prepare_env_config(
                args.env_config,
                tasks=[task_name],
                num_agents=agent_count,
                scaling_mode=args.scaling_mode,
            )
            methods = {}
            for method_name in ["random", "heuristic"]:
                env = CentralizedMultiUAVEnv(env_config)
                policy = make_baseline(method_name, env_config)
                episode_records = []
                component_totals = []
                for episode_idx in range(args.episodes):
                    obs, _ = env.reset(seed=int(env_config.get("seed", 0)) + episode_idx)
                    done = False
                    total_reward = 0.0
                    component_accumulator = {}
                    info = {}
                    while not done:
                        action = policy.act(obs)
                        obs, reward, terminated, truncated, info = env.step(action)
                        total_reward += reward
                        for key, value in info.get("reward_components", {}).items():
                            component_accumulator[key] = component_accumulator.get(key, 0.0) + float(value)
                        done = terminated or truncated
                    record = {
                        "task_name": task_name,
                        "num_agents": agent_count,
                        "scaling_mode": args.scaling_mode,
                        "method": method_name,
                        "return": float(total_reward),
                        "success": float(info.get("success", False)),
                        "collision_rate": float(info.get("collision_rate", 0.0)),
                        "path_length": float(info.get("path_length", 0.0)),
                    }
                    record.update({f"component_{key}": float(value) for key, value in component_accumulator.items()})
                    episode_records.append(record)
                    component_totals.append(component_accumulator)
                methods[method_name] = episode_records
                component_keys = sorted({key for totals in component_totals for key in totals.keys()})
                row = {
                    "task_name": task_name,
                    "num_agents": agent_count,
                    "scaling_mode": args.scaling_mode,
                    "method": method_name,
                    "return_mean": float(np.mean([r["return"] for r in episode_records])),
                    "return_std": float(np.std([r["return"] for r in episode_records])),
                    "success_rate": float(np.mean([r["success"] for r in episode_records])),
                    "collision_rate": float(np.mean([r["collision_rate"] for r in episode_records])),
                    "path_length": float(np.mean([r["path_length"] for r in episode_records])),
                }
                for component_key in component_keys:
                    values = [record.get(f"component_{component_key}", 0.0) for record in episode_records]
                    row[f"{component_key}_mean"] = float(np.mean(values))
                    row[f"{component_key}_std"] = float(np.std(values))
                stats.append(row)

            all_components = sorted(
                {
                    key
                    for method_records in methods.values()
                    for record in method_records
                    for key in record.keys()
                    if key.startswith("component_")
                }
            )
            if all_components:
                fig, ax = plt.subplots(figsize=(10, 4))
                x = np.arange(len(all_components))
                width = 0.35
                for offset, method_name in enumerate(["random", "heuristic"]):
                    means = [np.mean([record.get(component, 0.0) for record in methods[method_name]]) for component in all_components]
                    ax.bar(x + offset * width, means, width=width, label=method_name)
                ax.set_xticks(x + width / 2)
                ax.set_xticklabels([component.replace("component_", "") for component in all_components], rotation=30, ha="right")
                ax.set_title(f"{task_name} | N={agent_count}")
                ax.legend()
                fig.tight_layout()
                fig.savefig(plot_dir / f"{task_name}_N{agent_count}.png", dpi=150)
                plt.close(fig)

    write_metrics_csv(stats, "outputs/smoke/diagnostics/reward_stats.csv")
    print("reward_diagnostics=outputs/smoke/diagnostics/reward_stats.csv")


if __name__ == "__main__":
    main()
