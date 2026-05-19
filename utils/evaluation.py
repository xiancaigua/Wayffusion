from __future__ import annotations

import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
import torch

from envs.metrics import compute_reference_normalization
from policies import observation_to_tensor
from utils.profiling import get_memory_usage_mb, measure_policy_latency_ms


def flatten_reward_components(info: dict) -> dict[str, float]:
    components = dict(info.get("reward_components", {}))
    return {f"reward_{key}": float(value) for key, value in components.items()}


def compute_reference_table(records: Iterable[dict]) -> dict[tuple[str, int, str], dict[str, float]]:
    grouped: dict[tuple[str, int, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        key = (str(record["task_name"]), int(record["num_agents"]), str(record.get("scaling_mode", "fixed_map")))
        return_key = "return_mean" if "return_mean" in record else "return"
        grouped[key][str(record["method"])].append(float(record[return_key]))
    table: dict[tuple[str, int, str], dict[str, float]] = {}
    for key, values in grouped.items():
        table[key] = {
            "random_return": float(np.mean(values.get("random", [0.0]))),
            "heuristic_return": float(np.mean(values.get("heuristic", [1.0]))),
        }
    return table


def apply_reference_normalization(record: dict, reference_table: dict[tuple[str, int, str], dict[str, float]]) -> dict:
    key = (str(record["task_name"]), int(record["num_agents"]), str(record.get("scaling_mode", "fixed_map")))
    refs = reference_table.get(key)
    record = dict(record)
    if "return" in record and "raw_return" not in record:
        record["raw_return"] = float(record["return"])
    if refs is None:
        return record
    return_key = "return_mean" if "return_mean" in record else "return"
    record.update(
        compute_reference_normalization(
            float(record[return_key]),
            refs["random_return"],
            refs["heuristic_return"],
        )
    )
    return record


def aggregate_episode_records(records: list[dict]) -> dict:
    if not records:
        return {}
    summary = {}
    keys = {key for record in records for key in record}
    for key in keys:
        values = [record[key] for record in records if key in record]
        if not values:
            continue
        if isinstance(values[0], (int, float, np.floating, np.integer)):
            numeric_values = [float(value) for value in values]
            summary[f"{key}_mean"] = float(np.mean(numeric_values))
            summary[f"{key}_std"] = float(np.std(numeric_values))
        else:
            summary[key] = values[0] if all(value == values[0] for value in values[1:]) else "mixed"
    return summary


def _sanitize_media_tag(value: str) -> str:
    keep = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "episode"


def _write_episode_media(
    frames: list[np.ndarray],
    output_dir: Path,
    prefix: str,
    media_format: str,
    fps: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    media_format = str(media_format).lower()
    fps = max(int(fps), 1)
    safe_prefix = _sanitize_media_tag(prefix)
    prepared_frames = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    if media_format == "mp4":
        output_path = output_dir / f"{safe_prefix}.mp4"
        try:
            with imageio.get_writer(output_path, fps=fps) as writer:
                for frame in prepared_frames:
                    writer.append_data(frame)
            return output_path
        except Exception:
            media_format = "gif"
    output_path = output_dir / f"{safe_prefix}.gif"
    imageio.mimsave(output_path, prepared_frames, duration=1.0 / float(fps), loop=0)
    return output_path


def evaluate_policy_episodes(
    env,
    policy,
    episodes: int,
    device: torch.device,
    deterministic: bool = True,
    headless: bool = True,
    record_dir: str | Path | None = None,
    record_episodes: int = 0,
    record_format: str = "gif",
    record_fps: int = 8,
    record_prefix: str = "episode",
) -> list[dict]:
    records = []
    record_root = Path(record_dir) if record_dir is not None else None
    if record_root is not None:
        record_root.mkdir(parents=True, exist_ok=True)
    record_limit = max(int(record_episodes), 0)
    for episode_idx in range(episodes):
        episode_seed = int(env.config.get("seed", 0)) + 1000 + episode_idx
        observation, reset_info = env.reset(seed=episode_seed)
        done = False
        total_reward = 0.0
        step_count = 0
        inference_times = []
        rollout_start = time.perf_counter()
        info = dict(reset_info)
        task_name = str(info.get("task_name", getattr(getattr(env, "current_task", None), "name", "episode")))
        should_record = record_root is not None and episode_idx < record_limit
        frames: list[np.ndarray] = []
        if not headless:
            env.render(mode="human")
        if should_record:
            frames.append(np.asarray(env.render(mode="rgb_array"), dtype=np.uint8))
        while not done:
            obs_tensor = observation_to_tensor(observation, device)
            inference_times.append(measure_policy_latency_ms(policy, obs_tensor, repeats=1))
            with torch.no_grad():
                if hasattr(policy, "act_deterministic"):
                    action_tensor = policy.act_deterministic(obs_tensor)
                else:
                    action_tensor, _, _, _ = policy.get_action_and_value(obs_tensor)
            action = action_tensor.squeeze(0).cpu().numpy()
            if not deterministic:
                action = np.clip(action + np.random.normal(scale=0.1, size=action.shape), -1.0, 1.0)
            observation, reward, terminated, truncated, info = env.step(action)
            if not headless:
                env.render(mode="human")
            if should_record:
                frames.append(np.asarray(env.render(mode="rgb_array"), dtype=np.uint8))
            total_reward += reward
            step_count += 1
            done = terminated or truncated
        wall_clock = time.perf_counter() - rollout_start
        recording_path = None
        if should_record and frames:
            recording_path = _write_episode_media(
                frames,
                record_root,
                f"{record_prefix}_ep{episode_idx:03d}_{task_name}_N{info.get('num_agents', env.config['num_agents'])}",
                record_format,
                record_fps,
            )
        record = {
            "episode": episode_idx,
            "return": float(total_reward),
            "raw_return": float(total_reward),
            "success_rate": float(info.get("success", False)),
            "collision_rate": float(info.get("collision_rate", 0.0)),
            "path_length": float(info.get("path_length", 0.0)),
            "inference_latency_ms": float(np.mean(inference_times)) if inference_times else 0.0,
            "rollout_steps_per_sec": float(step_count / max(wall_clock, 1e-6)),
            "wall_clock_time": float(wall_clock),
            "memory_usage_mb": get_memory_usage_mb(),
            "intrinsic_score": float(info.get("intrinsic_score", info.get("normalized_score", 0.0))),
            "task_name": info.get("task_name"),
            "num_agents": int(info.get("num_agents", env.config["num_agents"])),
            "scaling_mode": info.get("scaling_mode", env.config.get("scaling_mode", "fixed_map")),
            **flatten_reward_components(info),
        }
        if recording_path is not None:
            record["recording_path"] = str(recording_path)
        for key, value in info.get("task_specific_metrics", {}).items():
            if isinstance(value, (int, float, np.floating)):
                record[key] = float(value)
        records.append(record)
    return records


def make_fixed_task_eval_config(base_env_config: dict, task_name: str) -> dict:
    from tasks import TASK_ORDER

    task_name = str(task_name)
    eval_config = deepcopy(base_env_config)
    eval_config["task_name"] = task_name
    eval_config["task_names"] = [task_name]
    eval_config["task_sampling_probs"] = {
        name: (1.0 if name == task_name else 0.0) for name in TASK_ORDER
    }
    return eval_config


def evaluate_policy_per_task(
    base_env_config: dict,
    policy,
    task_names: list[str],
    episodes_per_task: int,
    device,
    headless: bool = True,
    record_dir: Path | None = None,
    record_episodes: int = 0,
    record_format: str = "gif",
    record_fps: int = 8,
    record_prefix: str = "eval",
    normalize_with_reference: bool = False,
    reference_episodes: int | None = None,
) -> tuple[list[dict], dict[str, dict], dict]:
    from envs import CentralizedMultiUAVEnv

    all_records: list[dict] = []
    task_summaries: dict[str, dict] = {}
    task_names = [str(task_name) for task_name in task_names]
    per_task_episodes = max(int(episodes_per_task), 1)
    reference_episodes = int(reference_episodes or per_task_episodes)
    record_root = Path(record_dir) if record_dir is not None else None

    for task_name in task_names:
        eval_config = make_fixed_task_eval_config(base_env_config, task_name)
        env = CentralizedMultiUAVEnv(eval_config)
        task_record_dir = record_root / task_name if record_root is not None else None
        records = evaluate_policy_episodes(
            env,
            policy,
            per_task_episodes,
            device,
            deterministic=True,
            headless=headless,
            record_dir=task_record_dir,
            record_episodes=record_episodes,
            record_format=record_format,
            record_fps=record_fps,
            record_prefix=f"{record_prefix}_{task_name}",
        )
        if normalize_with_reference:
            from scripts._common import baseline_reference_table

            reference_table = baseline_reference_table(eval_config, episodes=reference_episodes)
            records = [apply_reference_normalization(record, reference_table) for record in records]
        for record in records:
            record["task_name"] = task_name
            record["eval_group"] = task_name
        task_summary = aggregate_episode_records(records)
        task_summary["task_name"] = task_name
        task_summary["eval_group"] = task_name
        task_summaries[task_name] = task_summary
        all_records.extend(records)

    overall_summary = aggregate_episode_records(all_records)
    overall_summary["task_name"] = "overall"
    overall_summary["eval_group"] = "overall"
    return all_records, task_summaries, overall_summary


def flatten_task_eval_summaries(
    task_summaries: dict[str, dict],
    overall_summary: dict,
    prefix: str = "eval",
) -> dict[str, float]:
    flattened: dict[str, float] = {}

    def add_summary(summary_prefix: str, summary: dict) -> None:
        for key, value in summary.items():
            if not isinstance(value, (int, float, np.integer, np.floating)):
                continue
            if not key.endswith("_mean"):
                continue
            metric_name = key[: -len("_mean")]
            flattened[f"{summary_prefix}_{metric_name}"] = float(value)

    for task_name, summary in task_summaries.items():
        add_summary(f"{prefix}_{_sanitize_media_tag(task_name)}", summary)
    add_summary(f"{prefix}_overall", overall_summary)

    overall_aliases = {
        f"{prefix}_reward": flattened.get(f"{prefix}_overall_return"),
        f"{prefix}_success_rate": flattened.get(f"{prefix}_overall_success_rate"),
        f"{prefix}_collision_rate": flattened.get(f"{prefix}_overall_collision_rate"),
        f"{prefix}_path_length": flattened.get(f"{prefix}_overall_path_length"),
        f"{prefix}_inference_latency_ms": flattened.get(f"{prefix}_overall_inference_latency_ms"),
    }
    for key, value in overall_aliases.items():
        if value is not None:
            flattened[key] = float(value)
    return flattened
