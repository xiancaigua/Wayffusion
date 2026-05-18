from __future__ import annotations

import random
import time
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from policies import observation_to_tensor
from utils.profiling import get_memory_usage_mb, measure_policy_latency_ms


def _flatten_obs_numpy(observation: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([observation["task_field"].reshape(-1), observation["agents"].reshape(-1), observation["task_id"].reshape(-1), observation["global_info"].reshape(-1)], axis=0).astype(np.float32)


def _flatten_obs_tensor(observation: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [
            observation["task_field"].reshape(observation["task_field"].shape[0], -1),
            observation["agents"].reshape(observation["agents"].shape[0], -1),
            observation["task_id"].reshape(observation["task_id"].shape[0], -1),
            observation["global_info"].reshape(observation["global_info"].shape[0], -1),
        ],
        dim=-1,
    )


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.storage = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.storage.append(
            (
                {k: np.asarray(v, dtype=np.float32).copy() for k, v in obs.items()},
                np.asarray(action, dtype=np.float32).copy(),
                float(reward),
                {k: np.asarray(v, dtype=np.float32).copy() for k, v in next_obs.items()},
                float(done),
            )
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.storage, batch_size)
        obs = {key: np.stack([item[0][key] for item in batch], axis=0).astype(np.float32) for key in batch[0][0]}
        actions = np.stack([item[1] for item in batch], axis=0).astype(np.float32)
        rewards = np.asarray([item[2] for item in batch], dtype=np.float32)
        next_obs = {key: np.stack([item[3][key] for item in batch], axis=0).astype(np.float32) for key in batch[0][3]}
        dones = np.asarray([item[4] for item in batch], dtype=np.float32)
        return obs, actions, rewards, next_obs, dones

    def __len__(self) -> int:
        return len(self.storage)


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action.reshape(action.shape[0], -1)], dim=-1)
        return self.net(x).squeeze(-1)


class TD3Trainer:
    def __init__(self, env_batch, actor: nn.Module, train_config: dict, device: str | None = None):
        self.env_batch = env_batch
        self.actor = actor
        self.train_config = train_config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor.to(self.device)
        sample_obs, _ = self.env_batch.reset()
        self.current_obs = sample_obs
        state_dim = _flatten_obs_numpy({k: v[0] for k, v in sample_obs.items()}).shape[0]
        action_dim = int(np.prod(self.env_batch.envs[0].action_space.shape))
        self.q1 = QNetwork(state_dim + action_dim).to(self.device)
        self.q2 = QNetwork(state_dim + action_dim).to(self.device)
        self.target_q1 = QNetwork(state_dim + action_dim).to(self.device)
        self.target_q2 = QNetwork(state_dim + action_dim).to(self.device)
        self.target_actor = deepcopy(actor)
        self.target_actor.to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(train_config["learning_rate"]))
        self.critic_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=float(train_config["learning_rate"]))
        self.replay = ReplayBuffer(int(train_config["replay_size"]))
        self.total_steps = 0
        self.update_step = 0

    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def _sample_random_actions(self, batch_size: int, action_shape: tuple[int, ...]) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, size=(batch_size, *action_shape)).astype(np.float32)

    def _collect_step(self) -> dict:
        if self.total_steps < int(self.train_config["warmup_steps"]):
            actions = self._sample_random_actions(self.env_batch.num_envs, self.env_batch.envs[0].action_space.shape)
            latency = 0.0
        else:
            obs_tensor = observation_to_tensor(self.current_obs, self.device, already_batched=True)
            latency = measure_policy_latency_ms(self.actor, obs_tensor, repeats=1)
            with torch.no_grad():
                actions = self.actor.act_deterministic(obs_tensor).cpu().numpy()
        next_obs, rewards, dones, truncs, infos = self.env_batch.step(actions)
        for env_idx in range(self.env_batch.num_envs):
            obs_single = {key: self.current_obs[key][env_idx] for key in self.current_obs}
            next_single = {key: next_obs[key][env_idx] for key in next_obs}
            self.replay.add(obs_single, actions[env_idx], rewards[env_idx], next_single, dones[env_idx])
        self.current_obs = next_obs
        self.total_steps += self.env_batch.num_envs
        return {"rollout_reward": float(rewards.mean()), "inference_latency_ms": float(latency)}

    def _update_once(self) -> dict:
        obs, actions, rewards, next_obs, dones = self.replay.sample(int(self.train_config["batch_size"]))
        obs_t = {key: torch.as_tensor(value, dtype=torch.float32, device=self.device) for key, value in obs.items()}
        next_obs_t = {key: torch.as_tensor(value, dtype=torch.float32, device=self.device) for key, value in next_obs.items()}
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        state = _flatten_obs_tensor(obs_t)
        next_state = _flatten_obs_tensor(next_obs_t)
        with torch.no_grad():
            noise = torch.randn_like(actions_t) * float(self.train_config["policy_noise"])
            noise = torch.clamp(noise, -float(self.train_config["noise_clip"]), float(self.train_config["noise_clip"]))
            next_action = torch.clamp(self.target_actor.act_deterministic(next_obs_t) + noise, -1.0, 1.0)
            target_q = torch.min(self.target_q1(next_state, next_action), self.target_q2(next_state, next_action))
            target = rewards_t + float(self.train_config["gamma"]) * (1.0 - dones_t) * target_q
        q1_loss = ((self.q1(state, actions_t) - target) ** 2).mean()
        q2_loss = ((self.q2(state, actions_t) - target) ** 2).mean()
        critic_loss = q1_loss + q2_loss
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_value = 0.0
        self.update_step += 1
        if self.update_step % int(self.train_config["policy_delay"]) == 0:
            actor_loss = -self.q1(state, self.actor.act_deterministic(obs_t)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            tau = float(self.train_config["tau"])
            self._soft_update(self.target_q1, self.q1, tau)
            self._soft_update(self.target_q2, self.q2, tau)
            self._soft_update(self.target_actor, self.actor, tau)
            actor_loss_value = float(actor_loss.item())
        return {"actor_loss": actor_loss_value, "critic_loss": float(critic_loss.item())}

    def evaluate(
        self,
        env,
        episodes: int = 5,
        deterministic: bool = True,
        headless: bool = True,
        record_dir: str | Path | None = None,
        record_episodes: int = 0,
        record_format: str = "gif",
        record_fps: int = 8,
        record_prefix: str = "eval",
    ) -> dict:
        from utils import aggregate_episode_records, evaluate_policy_episodes

        records = evaluate_policy_episodes(
            env,
            self.actor,
            episodes,
            self.device,
            deterministic=deterministic,
            headless=headless,
            record_dir=record_dir,
            record_episodes=record_episodes,
            record_format=record_format,
            record_fps=record_fps,
            record_prefix=record_prefix,
        )
        summary = aggregate_episode_records(records)
        result = {
            "eval_reward": float(summary.get("return_mean", 0.0)),
            "eval_success_rate": float(summary.get("success_rate_mean", 0.0)),
            "eval_path_length": float(summary.get("path_length_mean", 0.0)),
            "eval_collision_rate": float(summary.get("collision_rate_mean", 0.0)),
            "eval_inference_latency_ms": float(summary.get("inference_latency_ms_mean", 0.0)),
        }
        if record_dir is not None and int(record_episodes) > 0:
            result["eval_media_dir"] = str(record_dir)
        return result

    def train(
        self,
        output_dir: str | Path,
        eval_env=None,
        eval_episodes: int = 5,
        headless: bool = True,
        record_eval_episodes: int = 0,
        record_format: str = "gif",
        record_fps: int = 8,
        record_interval: int = 1,
        log_callback: Callable[[dict], None] | None = None,
    ) -> list[dict]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        metrics = []
        start = time.perf_counter()
        total_steps_target = int(self.train_config["total_steps"])
        eval_env = eval_env or self.env_batch.envs[0]
        eval_interval_steps = max(int(self.train_config.get("eval_interval_steps", total_steps_target)), 1)
        next_eval_step = eval_interval_steps
        eval_count = 0
        record_interval = max(int(record_interval), 1)
        while self.total_steps < total_steps_target:
            rollout_stats = self._collect_step()
            record = {
                "step": self.total_steps,
                "rollout_reward": rollout_stats["rollout_reward"],
                "inference_latency_ms": rollout_stats["inference_latency_ms"],
                "memory_usage_mb": get_memory_usage_mb(),
            }
            if len(self.replay) >= int(self.train_config["batch_size"]):
                record.update(self._update_once())
            if self.total_steps >= next_eval_step or self.total_steps >= total_steps_target:
                eval_count += 1
                media_dir = None
                media_episodes = 0
                if int(record_eval_episodes) > 0 and eval_count % record_interval == 0:
                    media_dir = output_dir / "media" / f"step_{self.total_steps:08d}"
                    media_episodes = int(record_eval_episodes)
                record.update(
                    self.evaluate(
                        eval_env,
                        episodes=eval_episodes,
                        headless=headless,
                        record_dir=media_dir,
                        record_episodes=media_episodes,
                        record_format=record_format,
                        record_fps=record_fps,
                        record_prefix=f"step_{self.total_steps:08d}",
                    )
                )
                checkpoint_path = checkpoints_dir / f"checkpoint_{self.total_steps:08d}.pt"
                torch.save({"model_state_dict": self.actor.state_dict(), "train_config": self.train_config}, checkpoint_path)
                record["checkpoint_path"] = str(checkpoint_path)
                while next_eval_step <= self.total_steps:
                    next_eval_step += eval_interval_steps
            metrics.append(record)
            if log_callback is not None:
                log_callback(record)
        checkpoint = checkpoints_dir / "checkpoint_final.pt"
        torch.save({"model_state_dict": self.actor.state_dict(), "train_config": self.train_config}, checkpoint)
        final_record = {
            "step": self.total_steps,
            "checkpoint_path": str(checkpoint),
            "wall_clock_time": float(time.perf_counter() - start),
        }
        metrics.append(final_record)
        if log_callback is not None:
            log_callback(final_record)
        return metrics
