from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", required=True)
    parser.add_argument("--config", default="configs/policy/sac_cnn_deepsets.yaml")
    parser.add_argument("--env-config", default="configs/env/multitask.yaml")
    parser.add_argument("--agent_counts", nargs="+", type=int, default=[4])
    parser.add_argument("--scaling_mode", default="fixed_map")
    parser.add_argument("--obs_variant", default="multi_channel_field+task_id")
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--run_timestamp", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.groups, "r", encoding="utf-8") as handle:
        grouping = yaml.safe_load(handle) or {}
    groups = grouping.get("groups") or []
    if not groups:
        raise ValueError(f"No groups found in {args.groups}")
    for group_idx, tasks in enumerate(groups):
        run_name = f"mtrl_cg_group{group_idx}_{'_'.join(tasks)}"
        cmd = [
            args.python_bin,
            "scripts/train_sac.py",
            "--config",
            args.config,
            "--env-config",
            args.env_config,
            "--tasks",
            *tasks,
            "--agent_counts",
            *[str(n) for n in args.agent_counts],
            "--scaling_mode",
            args.scaling_mode,
            "--obs_variant",
            args.obs_variant,
            "--eval_episodes",
            str(args.eval_episodes),
            "--run_name",
            run_name,
            "--headless",
            "--tensorboard",
        ]
        if args.run_timestamp:
            cmd.extend(["--run_timestamp", args.run_timestamp])
        print(f"group={group_idx} run_name={run_name} command={' '.join(cmd)}", flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
