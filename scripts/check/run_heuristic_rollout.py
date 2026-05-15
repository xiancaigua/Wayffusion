from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts._common import build_baseline, build_env, ensure_dir, load_env_config, run_baseline_episode, save_rollout_artifacts, write_metrics_csv
except ModuleNotFoundError:
    from _common import build_baseline, build_env, ensure_dir, load_env_config, run_baseline_episode, save_rollout_artifacts, write_metrics_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="heuristic", choices=["heuristic", "random", "greedy_goal", "greedy_coverage", "geometric_formation", "risk_potential"])
    parser.add_argument("--task", default="goal_nav", choices=["goal_nav", "coverage", "formation", "risk_nav"])
    parser.add_argument("--config", default="configs/env/multitask.yaml")
    parser.add_argument("--output-dir", default="outputs/smoke/sanity")
    args = parser.parse_args()

    config = load_env_config(args.config, override={"task_name": args.task})
    env = build_env(config, task_name=args.task)
    policy = build_baseline(args.policy, config)
    _, info = run_baseline_episode(env, policy, seed=int(config.get("seed", 0)))
    save_rollout_artifacts(info, ensure_dir(args.output_dir), f"{args.task}_{args.policy}")
    write_metrics_csv([{**info, **info.get("task_specific_metrics", {})}], ensure_dir(args.output_dir) / f"{args.task}_{args.policy}_metrics.csv")
    print({key: value for key, value in info.items() if key not in {"trajectory_history", "full_task_field", "per_agent_positions", "current_waypoints", "task_targets", "task_specific_metrics"}})


if __name__ == "__main__":
    main()
