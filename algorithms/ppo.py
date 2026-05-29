from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from policies import observation_to_tensor
from utils.profiling import get_memory_usage_mb, measure_policy_latency_ms


class RunningMeanStd:
    """Online scalar normalizer used for PPO reward normalization."""

    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, values: np.ndarray) -> None:
        """Merge one rollout reward batch into the running mean/variance."""

        values = np.asarray(values, dtype=np.float32)
        batch_mean = float(values.mean())
        batch_var = float(values.var())
        batch_count = values.size
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta * delta * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / np.sqrt(self.var + 1e-8)


@dataclass
class PPOBatch:
    """One PPO rollout after GAE/return computation.

    Time and vector-env dimensions are kept separate here. The update step later
    flattens them into a single batch dimension for shuffled minibatches.
    """

    observations: dict[str, np.ndarray]
    actions: np.ndarray
    logprobs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


class RolloutBuffer:
    """Temporary storage for one on-policy rollout horizon."""

    def __init__(self):
        self.obs: list[dict[str, np.ndarray]] = []
        self.actions: list[np.ndarray] = []
        self.logprobs: list[np.ndarray] = []
        self.rewards: list[np.ndarray] = []
        self.dones: list[np.ndarray] = []
        self.values: list[np.ndarray] = []

    def add(self, obs, action, logprob, reward, done, value):
        # Copy arrays at collection time because vector environments reuse and
        # mutate their current observation buffers between steps.
        self.obs.append({k: np.asarray(v, dtype=np.float32).copy() for k, v in obs.items()})
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.logprobs.append(np.asarray(logprob, dtype=np.float32).copy())
        self.rewards.append(np.asarray(reward, dtype=np.float32).copy())
        self.dones.append(np.asarray(done, dtype=np.float32).copy())
        self.values.append(np.asarray(value, dtype=np.float32).copy())

    def build(self, advantages: np.ndarray, returns: np.ndarray) -> PPOBatch:
        obs_dict = {key: np.stack([obs[key] for obs in self.obs], axis=0).astype(np.float32) for key in self.obs[0]}
        return PPOBatch(
            observations=obs_dict,
            actions=np.stack(self.actions, axis=0).astype(np.float32),
            logprobs=np.stack(self.logprobs, axis=0).astype(np.float32),
            rewards=np.stack(self.rewards, axis=0).astype(np.float32),
            dones=np.stack(self.dones, axis=0).astype(np.float32),
            values=np.stack(self.values, axis=0).astype(np.float32),
            advantages=advantages.astype(np.float32),
            returns=returns.astype(np.float32),
        )


class PPOTrainer:
    """Minimal PPO trainer for centralized multi-UAV policies.

    The trainer is intentionally framework-light: it expects a synchronous
    vectorized environment batch, a policy implementing `get_action_and_value`,
    and config fields for rollout length, GAE, clipping, and optimizer settings.
    """

    def __init__(self, env_batch, policy: nn.Module, train_config: dict, device: str | None = None):
        self.env_batch = env_batch
        self.policy = policy
        self.train_config = train_config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(train_config["learning_rate"]))
        self.reward_rms = RunningMeanStd()
        self.current_obs, _ = self.env_batch.reset()
        self.global_step = 0
        self.completed_episodes = 0
        self._clamp_policy_log_std()

    def set_env_batch(self, env_batch) -> None:
        """Swap vectorized envs for variable-N training and reset observations."""

        self.env_batch = env_batch
        self.current_obs, _ = self.env_batch.reset()

    def _flatten_time_env(self, array: np.ndarray) -> np.ndarray:
        """Collapse [time, env, ...] into [time * env, ...] for minibatches."""

        return array.reshape(array.shape[0] * array.shape[1], *array.shape[2:])

    def _flatten_obs(self, observations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: self._flatten_time_env(value) for key, value in observations.items()}

    def _set_lr(self, update_idx: int, total_updates: int) -> None:
        """Apply optional linear learning-rate annealing across PPO updates."""

        if self.train_config.get("lr_schedule", "constant") != "linear":
            return
        frac = 1.0 - ((update_idx - 1.0) / max(total_updates, 1))
        lr = float(self.train_config["learning_rate"]) * frac
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _clamp_policy_log_std(self) -> None:
        if not hasattr(self.policy, "log_std"):
            return
        log_std_min = float(self.train_config.get("log_std_min", getattr(self.policy, "log_std_min", -1.5)))
        log_std_max = float(self.train_config.get("log_std_max", getattr(self.policy, "log_std_max", 0.5)))
        with torch.no_grad():
            self.policy.log_std.clamp_(log_std_min, log_std_max)

    def collect_rollout(self) -> tuple[PPOBatch, dict]:
        """Collect one on-policy rollout and compute GAE advantages.

        PPO requires fresh samples from the current policy. For each environment
        step we store the sampled action, its old log-prob, the critic value, and
        the normalized reward. After the horizon, a final critic bootstrap value
        is used to compute generalized advantage estimates backward in time.
        """

        horizon = int(self.train_config["rollout_steps"])
        gamma = float(self.train_config["gamma"])
        gae_lambda = float(self.train_config["gae_lambda"])
        buffer = RolloutBuffer()
        rollout_rewards = []
        rollout_norm_rewards = []
        rollout_action_abs = []
        rollout_action_saturation = []
        inference_latencies = []
        terminal_successes = []
        terminal_goal_coverages = []
        terminal_collision_rates = []
        terminal_path_lengths = []
        start = time.perf_counter()
        completed_episodes = 0
        for _ in range(horizon):
            obs_tensor = observation_to_tensor(self.current_obs, self.device, already_batched=True)
            inference_latencies.append(measure_policy_latency_ms(self.policy, obs_tensor, repeats=1))
            with torch.no_grad():
                action_tensor, logprob_tensor, _, value_tensor = self.policy.get_action_and_value(obs_tensor)
            actions = action_tensor.cpu().numpy()
            rollout_action_abs.append(float(np.abs(actions).mean()))
            rollout_action_saturation.append(float((np.abs(actions) >= 0.95).mean()))
            next_obs, rewards, dones, truncs, infos = self.env_batch.step(actions)
            completed_episodes += int(np.count_nonzero(dones))
            for info in infos:
                terminal_info = info.get("terminal_info") if isinstance(info, dict) else None
                if terminal_info is None:
                    continue
                terminal_successes.append(float(terminal_info.get("success", False)))
                terminal_goal_coverages.append(float(terminal_info.get("goal_coverage_ratio", terminal_info.get("coverage_ratio", 0.0))))
                terminal_collision_rates.append(float(terminal_info.get("collision_rate", 0.0)))
                terminal_path_lengths.append(float(terminal_info.get("path_length", 0.0)))
            norm_rewards = rewards.copy()
            if self.train_config.get("reward_norm", True):
                self.reward_rms.update(rewards)
                norm_rewards = self.reward_rms.normalize(rewards)
            buffer.add(
                self.current_obs,
                actions,
                logprob_tensor.cpu().numpy(),
                norm_rewards,
                dones,
                value_tensor.cpu().numpy(),
            )
            rollout_rewards.append(rewards.mean())
            rollout_norm_rewards.append(norm_rewards.mean())
            self.current_obs = next_obs
            self.global_step += self.env_batch.num_envs
        self.completed_episodes += completed_episodes
        with torch.no_grad():
            # Bootstrap from the critic at the observation following the rollout.
            # This keeps truncated rollouts from treating the horizon as zero
            # value when the episode is still alive.
            next_value = self.policy.get_action_and_value(
                observation_to_tensor(self.current_obs, self.device, already_batched=True)
            )[3].cpu().numpy()

        rewards = np.asarray(buffer.rewards, dtype=np.float32)
        dones = np.asarray(buffer.dones, dtype=np.float32)
        values = np.concatenate([np.asarray(buffer.values, dtype=np.float32), next_value[None, :]], axis=0)
        advantages = np.zeros_like(rewards)
        gae = np.zeros((rewards.shape[1],), dtype=np.float32)
        for t in reversed(range(horizon)):
            # GAE recursively mixes one-step TD error with longer-horizon
            # estimates. done masks stop credit assignment across episode resets.
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        elapsed = time.perf_counter() - start
        reward_values = np.asarray(rollout_rewards, dtype=np.float32)
        norm_reward_values = np.asarray(rollout_norm_rewards, dtype=np.float32)
        return buffer.build(advantages, returns), {
            "mean_rollout_reward": float(np.mean(reward_values)),
            "rollout_reward_std": float(np.std(reward_values)),
            "mean_normalized_rollout_reward": float(np.mean(norm_reward_values)),
            "normalized_rollout_reward_std": float(np.std(norm_reward_values)),
            "reward_rms_mean": float(self.reward_rms.mean),
            "reward_rms_std": float(np.sqrt(self.reward_rms.var + 1e-8)),
            "action_abs_mean": float(np.mean(rollout_action_abs)) if rollout_action_abs else 0.0,
            "action_saturation_frac": float(np.mean(rollout_action_saturation)) if rollout_action_saturation else 0.0,
            "rollout_episode_success_rate": float(np.mean(terminal_successes)) if terminal_successes else 0.0,
            "rollout_terminal_goal_coverage": float(np.mean(terminal_goal_coverages)) if terminal_goal_coverages else 0.0,
            "rollout_terminal_collision_rate": float(np.mean(terminal_collision_rates)) if terminal_collision_rates else 0.0,
            "rollout_terminal_path_length": float(np.mean(terminal_path_lengths)) if terminal_path_lengths else 0.0,
            "rollout_steps_per_sec": float(horizon * self.env_batch.num_envs / max(elapsed, 1e-6)),
            "rollout_wall_clock_time": float(elapsed),
            "memory_usage_mb": get_memory_usage_mb(),
            "inference_latency_ms": float(np.mean(inference_latencies)) if inference_latencies else 0.0,
            "episodes_completed": int(completed_episodes),
            "cumulative_episodes": int(self.completed_episodes),
        }

    def update(self, batch: PPOBatch) -> dict:
        """Run clipped PPO optimization over one collected rollout.

        The old log-probabilities in the batch are fixed targets from rollout
        collection. The update recomputes log-probs under the current policy and
        clips the probability ratio so a single minibatch cannot move the policy
        too far from the behavior policy that generated the data.
        """

        advantages = batch.advantages.copy()
        if self.train_config.get("advantage_norm", True):
            advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-8)
        flat_obs = self._flatten_obs(batch.observations)
        flat_actions = self._flatten_time_env(batch.actions)
        flat_logprobs = batch.logprobs.reshape(-1)
        flat_returns = batch.returns.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        batch_size = flat_actions.shape[0]
        epochs = int(self.train_config["epochs"])
        minibatch_size = int(self.train_config["minibatch_size"])
        target_kl = float(self.train_config.get("target_kl", 0.0) or 0.0)
        raw_advantages = batch.advantages.reshape(-1)
        value_predictions = batch.values.reshape(-1)
        return_targets = batch.returns.reshape(-1)
        return_var = float(np.var(return_targets))
        explained_variance = 0.0
        if return_var > 1e-8:
            explained_variance = float(1.0 - np.var(return_targets - value_predictions) / return_var)
        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "ratio_mean": 0.0,
            "grad_norm": 0.0,
        }
        step_count = 0
        kl_early_stop = 0.0
        for _ in range(epochs):
            indices = np.random.permutation(batch_size)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                obs_tensors = {
                    key: torch.as_tensor(value[mb_idx], dtype=torch.float32, device=self.device)
                    for key, value in flat_obs.items()
                }
                actions = torch.as_tensor(flat_actions[mb_idx], dtype=torch.float32, device=self.device)
                old_logprob = torch.as_tensor(flat_logprobs[mb_idx], dtype=torch.float32, device=self.device)
                mb_advantages = torch.as_tensor(flat_advantages[mb_idx], dtype=torch.float32, device=self.device)
                mb_returns = torch.as_tensor(flat_returns[mb_idx], dtype=torch.float32, device=self.device)

                _, new_logprob, entropy, value = self.policy.get_action_and_value(obs_tensors, actions)
                # PPO surrogate objective: ratio > 1 means the current policy
                # makes the sampled action more likely than the rollout policy.
                logratio = new_logprob - old_logprob
                ratio = torch.exp(logratio)
                clip_coef = float(self.train_config["clip_coef"])
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                # Actor, critic, and entropy terms are kept explicit so logged
                # metrics map directly to the PPO paper's components.
                value_loss = 0.5 * ((value - mb_returns) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss + float(self.train_config["vf_coef"]) * value_loss - float(self.train_config["ent_coef"]) * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), float(self.train_config["max_grad_norm"]))
                self.optimizer.step()
                self._clamp_policy_log_std()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > clip_coef).float().mean()
                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy_loss.item())
                stats["approx_kl"] += float(approx_kl.item())
                stats["clip_frac"] += float(clip_frac.item())
                stats["ratio_mean"] += float(ratio.mean().item())
                stats["grad_norm"] += float(grad_norm.item())
                step_count += 1
                if target_kl > 0.0 and float(approx_kl.item()) > target_kl:
                    kl_early_stop = 1.0
                    break
            if kl_early_stop:
                break
        result = {key: value / max(step_count, 1) for key, value in stats.items()}
        result.update(
            {
                "advantage_mean": float(np.mean(raw_advantages)),
                "advantage_std": float(np.std(raw_advantages)),
                "return_mean": float(np.mean(return_targets)),
                "return_std": float(np.std(return_targets)),
                "value_pred_mean": float(np.mean(value_predictions)),
                "value_pred_std": float(np.std(value_predictions)),
                "explained_variance": explained_variance,
                "kl_early_stop": kl_early_stop,
            }
        )
        if hasattr(self.policy, "log_std"):
            log_std = self.policy.log_std.detach().cpu().numpy()
            result["log_std_mean"] = float(np.mean(log_std))
            result["policy_std_mean"] = float(np.mean(np.exp(log_std)))
        return result

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
            self.policy,
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

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self._clamp_policy_log_std()

    def train(
        self,
        output_dir: str | Path,
        eval_env=None,
        eval_task_names: list[str] | None = None,
        eval_base_env_config: dict | None = None,
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
        metrics_history = []
        best_eval_success = float("-inf")
        best_eval_reward = float("-inf")
        total_updates = int(self.train_config["total_updates"])
        target_episodes = int(self.train_config.get("target_episodes", 0) or 0)
        eval_interval = int(self.train_config.get("eval_interval", total_updates))
        eval_env = eval_env or self.env_batch.envs[0]
        eval_count = 0
        record_interval = max(int(record_interval), 1)
        for update_idx in range(1, total_updates + 1):
            self._set_lr(update_idx, total_updates)
            batch, rollout_stats = self.collect_rollout()
            train_stats = self.update(batch)
            record = {
                "update": update_idx,
                "wall_clock_time": rollout_stats["rollout_wall_clock_time"],
                **rollout_stats,
                **train_stats,
            }
            if update_idx % eval_interval == 0 or update_idx == total_updates:
                eval_count += 1
                media_dir = None
                media_episodes = 0
                if int(record_eval_episodes) > 0 and eval_count % record_interval == 0:
                    media_dir = output_dir / "media" / f"eval_{update_idx:04d}"
                    media_episodes = int(record_eval_episodes)
                if eval_task_names and eval_base_env_config is not None:
                    from utils import evaluate_policy_per_task, flatten_task_eval_summaries

                    _, task_summaries, overall_summary = evaluate_policy_per_task(
                        eval_base_env_config,
                        self.policy,
                        eval_task_names,
                        eval_episodes,
                        self.device,
                        headless=headless,
                        record_dir=media_dir,
                        record_episodes=media_episodes,
                        record_format=record_format,
                        record_fps=record_fps,
                        record_prefix=f"update_{update_idx:04d}",
                        normalize_with_reference=False,
                    )
                    record.update(flatten_task_eval_summaries(task_summaries, overall_summary, prefix="eval"))
                    if media_dir is not None and media_episodes > 0:
                        record["eval_media_dir"] = str(media_dir)
                else:
                    record.update(
                        self.evaluate(
                            eval_env,
                            episodes=eval_episodes,
                            headless=headless,
                            record_dir=media_dir,
                            record_episodes=media_episodes,
                            record_format=record_format,
                            record_fps=record_fps,
                            record_prefix=f"update_{update_idx:04d}",
                        )
                    )
                checkpoint_path = checkpoints_dir / f"checkpoint_{update_idx:04d}.pt"
                torch.save(
                    {"model_state_dict": self.policy.state_dict(), "train_config": self.train_config},
                    checkpoint_path,
                )
                record["checkpoint_path"] = str(checkpoint_path)
                eval_success = float(record.get("eval_success_rate", float("-inf")))
                eval_reward = float(record.get("eval_reward", float("-inf")))
                is_best = False
                if eval_success > best_eval_success:
                    is_best = True
                elif eval_success == best_eval_success and eval_reward > best_eval_reward:
                    is_best = True
                if is_best:
                    best_eval_success = eval_success
                    best_eval_reward = eval_reward
                    best_path = checkpoints_dir / "checkpoint_best_eval.pt"
                    torch.save(
                        {"model_state_dict": self.policy.state_dict(), "train_config": self.train_config},
                        best_path,
                    )
                    record["best_checkpoint_path"] = str(best_path)
                    summary_path = output_dir / "best_eval_summary.json"
                    summary_path.write_text(
                        json.dumps(
                            {
                                "update": int(update_idx),
                                "eval_success_rate": eval_success,
                                "eval_reward": eval_reward,
                                "checkpoint_path": str(best_path),
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
            metrics_history.append(record)
            if log_callback is not None:
                log_callback(record)
            if target_episodes > 0 and self.completed_episodes >= target_episodes:
                break
        return metrics_history
