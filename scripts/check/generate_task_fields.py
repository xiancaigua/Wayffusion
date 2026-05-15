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
    from scripts._common import build_baseline, build_env, ensure_dir, load_env_config, resolve_tasks, run_baseline_episode, save_rollout_artifacts, write_metrics_csv
except ModuleNotFoundError:
    from _common import build_baseline, build_env, ensure_dir, load_env_config, resolve_tasks, run_baseline_episode, save_rollout_artifacts, write_metrics_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env/multitask.yaml")
    parser.add_argument("--output-dir", default="outputs/smoke/sanity")
    args = parser.parse_args()

    config = load_env_config(args.config)
    policy = build_baseline("heuristic", config)
    output_dir = ensure_dir(args.output_dir)
    records = []
    for idx, task_name in enumerate(resolve_tasks(config)):
        env = build_env(config, task_name=task_name)
        _, info = run_baseline_episode(env, policy, seed=int(config.get("seed", 0)) + idx)
        prefix = f"{task_name}_heuristic"
        save_rollout_artifacts(info, output_dir, prefix)
        records.append({"task_name": task_name, **info, **info.get("task_specific_metrics", {})})
    write_metrics_csv(records, output_dir / "metrics.csv")


if __name__ == "__main__":
    main()
