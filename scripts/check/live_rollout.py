from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts._common import build_baseline, build_env, ensure_dir, load_env_config
except ModuleNotFoundError:
    from _common import build_baseline, build_env, ensure_dir, load_env_config


def draw_frame(ax, info: dict, step: int, total_reward: float, title_prefix: str) -> None:
    ax.clear()
    task_field = info["full_task_field"]
    map_size = float(info.get("map_size", 1.0))
    obstacle_map = task_field[0]
    risk_map = task_field[4]

    overlay = np.zeros((*risk_map.shape, 4), dtype=np.float32)
    overlay[..., 0] = risk_map
    overlay[..., 3] = np.clip(risk_map * 0.35, 0.0, 0.35)
    ax.imshow(overlay, origin="lower", extent=(0, map_size, 0, map_size))
    ax.imshow(obstacle_map, origin="lower", cmap="gray_r", alpha=0.45, extent=(0, map_size, 0, map_size))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    trajectories = info.get("trajectory_history", [])
    for idx, trajectory in enumerate(trajectories):
        trajectory = np.asarray(trajectory, dtype=np.float32)
        if trajectory.size == 0:
            continue
        color = colors[idx % len(colors)]
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=1.8)
        ax.scatter(trajectory[0, 0], trajectory[0, 1], color=color, marker="o", s=28)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color=color, marker="x", s=52)

    waypoints = np.asarray(info.get("current_waypoints", []), dtype=np.float32)
    for idx, waypoint in enumerate(waypoints):
        color = colors[idx % len(colors)]
        ax.scatter(waypoint[0], waypoint[1], color=color, marker="+", s=64)

    targets = np.asarray(info.get("task_targets", []), dtype=np.float32)
    if targets.size > 0:
        ax.scatter(targets[:, 0], targets[:, 1], color="gold", edgecolors="black", marker="*", s=120)

    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"{title_prefix} | step={step} | reward={total_reward:.2f} | "
        f"success={bool(info.get('success', False))} | collisions={info.get('collision_count', 0)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a live matplotlib rollout for a baseline UAV policy.")
    parser.add_argument("--config", default="configs/env/multitask.yaml")
    parser.add_argument("--task", default="goal_nav", choices=["goal_nav", "coverage", "formation", "risk_nav"])
    parser.add_argument(
        "--policy",
        default="heuristic",
        choices=["heuristic", "random", "greedy_goal", "greedy_coverage", "geometric_formation", "risk_potential"],
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.05, help="Seconds between rendered frames.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap below the environment max_steps.")
    parser.add_argument("--agent-count", type=int, default=None)
    parser.add_argument("--scaling-mode", default=None, choices=["fixed_map", "density_preserving"])
    parser.add_argument("--save-final", default=None, help="Optional path for the final frame PNG.")
    args = parser.parse_args()

    config = load_env_config(args.config)
    if args.agent_count is not None:
        config["num_agents"] = int(args.agent_count)
    if args.scaling_mode is not None:
        config["scaling_mode"] = args.scaling_mode
    if args.max_steps is not None:
        config["max_steps"] = int(args.max_steps)

    env = build_env(config, task_name=args.task)
    policy = build_baseline(args.policy, config)
    seed = int(config.get("seed", 0)) if args.seed is None else int(args.seed)
    observation, info = env.reset(seed=seed, options={"task_name": args.task})

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))
    title_prefix = f"{args.task} / {args.policy} / N={config['num_agents']}"
    total_reward = 0.0
    step = 0
    draw_frame(ax, info, step, total_reward, title_prefix)
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(max(args.delay, 0.001))

    done = False
    while not done and plt.fignum_exists(fig.number):
        action = policy.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        step += 1
        draw_frame(ax, info, step, total_reward, title_prefix)
        fig.tight_layout()
        plt.pause(max(args.delay, 0.001))
        done = bool(terminated or truncated)

    if args.save_final:
        output_path = Path(args.save_final)
        if not output_path.is_absolute():
            output_path = ensure_dir(ROOT / "outputs" / "smoke" / "live") / output_path
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"saved_final_frame={output_path}")

    print(
        f"task={args.task} policy={args.policy} steps={step} "
        f"reward={total_reward:.3f} success={bool(info.get('success', False))}"
    )
    if plt.fignum_exists(fig.number):
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
