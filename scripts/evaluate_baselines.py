from __future__ import annotations

import argparse
from copy import deepcopy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts._common import build_baseline, build_env, ensure_dir, load_env_config, load_generic_config, resolve_tasks, run_baseline_episode, save_rollout_artifacts, write_metrics_csv
except ModuleNotFoundError:
    from _common import build_baseline, build_env, ensure_dir, load_env_config, load_generic_config, resolve_tasks, run_baseline_episode, save_rollout_artifacts, write_metrics_csv

def run_single_task(eval_config: dict) -> list[dict]:
    env_config = load_env_config(eval_config["env_config"])
    records = []
    output_dir = ensure_dir(eval_config["output_dir"])
    for task_name in eval_config["tasks"]:
        for policy_name in eval_config["policies"]:
            env = build_env(env_config, task_name=task_name)
            policy = build_baseline(policy_name, env_config)
            best_info = None
            for episode_idx in range(int(eval_config["episodes_per_task"])):
                _, info = run_baseline_episode(env, policy, seed=int(env_config.get("seed", 0)) + episode_idx)
                record = {"task_name": task_name, "policy": policy_name, "episode": episode_idx, **info, **info.get("task_specific_metrics", {})}
                records.append(record)
                if best_info is None or info.get("normalized_score", 0.0) > best_info.get("normalized_score", 0.0):
                    best_info = deepcopy(info)
            if eval_config.get("save_rollouts", True) and best_info is not None:
                save_rollout_artifacts(best_info, output_dir / task_name, f"{policy_name}_best")
    return records


def run_multitask(eval_config: dict) -> list[dict]:
    env_config = load_env_config(eval_config["env_config"])
    records = []
    output_dir = ensure_dir(eval_config["output_dir"])
    for policy_name in eval_config["policies"]:
        env = build_env(env_config)
        policy = build_baseline(policy_name, env_config)
        best_by_task = {}
        for episode_idx in range(int(eval_config["episodes"])):
            _, info = run_baseline_episode(env, policy, seed=int(env_config.get("seed", 0)) + episode_idx)
            task_name = info["task_name"]
            record = {"task_name": task_name, "policy": policy_name, "episode": episode_idx, **info, **info.get("task_specific_metrics", {})}
            records.append(record)
            if task_name not in best_by_task or info["normalized_score"] > best_by_task[task_name]["normalized_score"]:
                best_by_task[task_name] = deepcopy(info)
        if eval_config.get("save_rollouts", True):
            for task_name, info in best_by_task.items():
                save_rollout_artifacts(info, output_dir / task_name, f"{policy_name}_best")
    return records


def run_generalization(eval_config: dict) -> list[dict]:
    base_config = load_env_config("configs/env/base.yaml")
    records = []
    output_dir = ensure_dir(eval_config["output_dir"])
    for split_name, split_override in [("train", eval_config["train_env_config"]), ("test", eval_config["test_env_config"])]:
        env_config = load_env_config("configs/env/multitask.yaml", override=split_override)
        for policy_name in eval_config["policies"]:
            env = build_env(env_config)
            policy = build_baseline(policy_name, env_config)
            split_records = []
            for episode_idx in range(int(eval_config["episodes"])):
                _, info = run_baseline_episode(env, policy, seed=int(base_config.get("seed", 0)) + 200 + episode_idx)
                split_records.append({"split": split_name, "policy": policy_name, "episode": episode_idx, **info, **info.get("task_specific_metrics", {})})
            records.extend(split_records)
            best_info = max(split_records, key=lambda item: item.get("normalized_score", 0.0))
            save_rollout_artifacts(best_info, output_dir / split_name, f"{policy_name}_{best_info['task_name']}_best")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval/eval_single_task.yaml")
    args = parser.parse_args()
    eval_config = load_generic_config(args.config)
    mode = eval_config["mode"]
    if mode == "single_task":
        records = run_single_task(eval_config)
    elif mode == "multitask":
        records = run_multitask(eval_config)
    elif mode == "generalization":
        records = run_generalization(eval_config)
    else:
        raise ValueError(f"Unsupported eval mode: {mode}")
    write_metrics_csv(records, ensure_dir(eval_config["output_dir"]) / "metrics.csv")


if __name__ == "__main__":
    main()
