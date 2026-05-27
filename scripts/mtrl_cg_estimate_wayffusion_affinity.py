from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mtrl_cg import cluster_from_affinity, write_grouping
from algorithms.sac import SACTrainer, _flatten_obs_tensor
from envs import CentralizedMultiUAVEnv
from policies import build_policy, observation_to_tensor
from scripts._common import (
    format_task_set_name,
    load_generic_config,
    normalize_task_names,
    observation_override_from_variant,
    prepare_env_config,
)
from utils import make_env_batch


def _collect_task_batch(env_config: dict, actor, device: torch.device, samples: int, seed: int) -> tuple[dict, np.ndarray, np.ndarray, dict, np.ndarray]:
    env = CentralizedMultiUAVEnv(deepcopy(env_config))
    obs, _ = env.reset(seed=seed)
    observations: list[dict[str, np.ndarray]] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    next_observations: list[dict[str, np.ndarray]] = []
    dones: list[float] = []
    while len(actions) < int(samples):
        obs_t = observation_to_tensor(obs, device)
        with torch.no_grad():
            action = actor.get_action_and_value(obs_t)[0].squeeze(0).cpu().numpy()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        observations.append({key: np.asarray(value, dtype=np.float32).copy() for key, value in obs.items()})
        actions.append(np.asarray(action, dtype=np.float32).copy())
        rewards.append(float(reward))
        next_observations.append({key: np.asarray(value, dtype=np.float32).copy() for key, value in next_obs.items()})
        done = bool(terminated or truncated)
        dones.append(float(done))
        obs = env.reset()[0] if done else next_obs
    obs_batch = {key: np.stack([item[key] for item in observations], axis=0).astype(np.float32) for key in observations[0]}
    next_obs_batch = {key: np.stack([item[key] for item in next_observations], axis=0).astype(np.float32) for key in next_observations[0]}
    return (
        obs_batch,
        np.stack(actions, axis=0).astype(np.float32),
        np.asarray(rewards, dtype=np.float32),
        next_obs_batch,
        np.asarray(dones, dtype=np.float32),
    )


def _critic_loss_for_batch(trainer: SACTrainer, batch) -> torch.Tensor:
    obs, actions, rewards, next_obs, dones = batch
    obs_t = {key: torch.as_tensor(value, dtype=torch.float32, device=trainer.device) for key, value in obs.items()}
    next_obs_t = {key: torch.as_tensor(value, dtype=torch.float32, device=trainer.device) for key, value in next_obs.items()}
    actions_t = torch.as_tensor(actions, dtype=torch.float32, device=trainer.device)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=trainer.device)
    dones_t = torch.as_tensor(dones, dtype=torch.float32, device=trainer.device)
    state = _flatten_obs_tensor(obs_t)
    next_state = _flatten_obs_tensor(next_obs_t)
    with torch.no_grad():
        next_action, next_logprob, _, _ = trainer.actor.get_action_and_value(next_obs_t)
        target_q = torch.min(trainer.target_q1(next_state, next_action), trainer.target_q2(next_state, next_action))
        target = rewards_t + float(trainer.train_config["gamma"]) * (1.0 - dones_t) * (target_q - trainer.alpha * next_logprob)
    return ((trainer.q1(state, actions_t) - target) ** 2).mean()


def _q_mean_for_batch(trainer: SACTrainer, batch) -> torch.Tensor:
    obs, actions, *_ = batch
    obs_t = {key: torch.as_tensor(value, dtype=torch.float32, device=trainer.device) for key, value in obs.items()}
    actions_t = torch.as_tensor(actions, dtype=torch.float32, device=trainer.device)
    return trainer.q1(_flatten_obs_tensor(obs_t), actions_t).mean()


def estimate_affinity(trainer: SACTrainer, task_batches: dict[str, tuple], learning_rate: float) -> np.ndarray:
    task_names = list(task_batches)
    params = [param for param in trainer.q1.parameters() if param.requires_grad]
    matrix = np.zeros((len(task_names), len(task_names)), dtype=np.float64)
    for i, source_task in enumerate(task_names):
        loss = _critic_loss_for_batch(trainer, task_batches[source_task])
        source_grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=True)
        for j, target_task in enumerate(task_names):
            if i == j:
                continue
            q_mean = _q_mean_for_batch(trainer, task_batches[target_task])
            target_grads = torch.autograd.grad(q_mean, params, retain_graph=False, create_graph=False, allow_unused=True)
            dot = torch.zeros((), dtype=torch.float32, device=trainer.device)
            for source_grad, target_grad in zip(source_grads, target_grads):
                if source_grad is None or target_grad is None:
                    continue
                dot = dot + (source_grad.detach() * target_grad.detach()).sum()
            denom = float(torch.clamp(q_mean.detach().abs(), min=1e-6).item())
            matrix[i, j] = float((-float(learning_rate) * dot / denom).detach().cpu().item())
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/policy/sac_server_smoke.yaml")
    parser.add_argument("--env-config", default="configs/env/multitask.yaml")
    parser.add_argument("--tasks", nargs="+", default=["goal_nav", "coverage", "formation", "risk_nav"])
    parser.add_argument("--agent_counts", nargs="+", type=int, default=[4])
    parser.add_argument("--scaling_mode", default="fixed_map")
    parser.add_argument("--obs_variant", default="multi_channel_field+task_id")
    parser.add_argument("--pretrain_steps", type=int, default=64)
    parser.add_argument("--estimate_interval", type=int, default=16)
    parser.add_argument("--affinity_batch", type=int, default=16)
    parser.add_argument("--num_groups", type=int, default=2)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    task_names = normalize_task_names(args.tasks)
    agent_count = int(args.agent_counts[0])
    train_config = load_generic_config(args.config)
    env_config = prepare_env_config(
        args.env_config,
        tasks=task_names,
        num_agents=agent_count,
        scaling_mode=args.scaling_mode,
        observation_override=observation_override_from_variant(args.obs_variant),
    )
    env_batch = make_env_batch(env_config, int(train_config.get("num_envs", 1)))
    policy = build_policy(train_config, env_batch.envs[0].observation_space, env_batch.envs[0].action_space)
    trainer = SACTrainer(env_batch, policy, train_config)
    matrices: list[np.ndarray] = []
    while trainer.total_steps < int(args.pretrain_steps):
        trainer._collect_step()
        if len(trainer.replay) >= int(train_config["batch_size"]):
            trainer._update_once()
        if trainer.total_steps % max(int(args.estimate_interval), 1) == 0:
            batches = {}
            for task_idx, task_name in enumerate(task_names):
                fixed_config = prepare_env_config(
                    args.env_config,
                    tasks=[task_name],
                    num_agents=agent_count,
                    scaling_mode=args.scaling_mode,
                    observation_override=observation_override_from_variant(args.obs_variant),
                )
                batches[task_name] = _collect_task_batch(
                    fixed_config,
                    trainer.actor,
                    trainer.device,
                    samples=int(args.affinity_batch),
                    seed=int(env_config.get("seed", 0)) + trainer.total_steps + task_idx,
                )
            matrices.append(estimate_affinity(trainer, batches, learning_rate=float(train_config["learning_rate"])))
            print(f"estimated_affinity_step={trainer.total_steps}", flush=True)

    if not matrices:
        raise RuntimeError("No affinity matrices were estimated; increase pretrain_steps or lower estimate_interval.")
    affinity = np.sum(matrices, axis=0)
    grouping = cluster_from_affinity(affinity, task_names=task_names, num_groups=int(args.num_groups))
    output_dir = Path(args.output_dir or (ROOT / "outputs" / "mtrl_cg" / f"{format_task_set_name(task_names)}_N{agent_count}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "affinity_raw.csv", affinity, delimiter=",")
    np.savetxt(output_dir / "affinity_symmetric.csv", np.asarray(grouping["affinity"]), delimiter=",")
    write_grouping(output_dir / "groups.yaml", grouping)
    with open(output_dir / "metadata.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump({"config": args.config, "env_config": args.env_config, "tasks": task_names, "agent_count": agent_count}, handle)
    print(f"mtrl_cg_affinity_dir={output_dir}")
    print(f"groups={grouping['groups']}")


if __name__ == "__main__":
    main()

