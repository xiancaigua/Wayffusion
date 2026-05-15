from __future__ import annotations

import time
from collections import defaultdict
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
    if refs is None:
        return record
    record = dict(record)
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
    scalar_keys = {
        key
        for record in records
        for key, value in record.items()
        if isinstance(value, (int, float, np.floating, np.integer))
    }
    for key in scalar_keys:
        values = [float(record[key]) for record in records if key in record]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
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
