from __future__ import annotations

import shutil
from pathlib import Path

import torch

from algorithms import PPOTrainer, SACTrainer, TD3Trainer
from policies import build_policy
from scripts._common import load_env_config, load_generic_config
from scripts.evaluate_scaling import evaluate_baseline_per_task
from utils import evaluate_policy_per_task, make_env_batch, make_fixed_task_eval_config


def _base_multitask_env_config() -> dict:
    return load_env_config(
        "configs/env/multitask.yaml",
        override={
            "task_name": None,
            "task_names": ["goal_nav", "coverage"],
            "task_sampling_probs": {"goal_nav": 0.5, "coverage": 0.5, "formation": 0.0, "risk_nav": 0.0},
            "num_agents": 4,
            "max_steps": 2,
            "seed": 11,
        },
    )


def _build_mlp_policy_for_config(env_config: dict):
    env_batch = make_env_batch(env_config, 1)
    policy_config = load_generic_config("configs/policy/ppo_mlp.yaml")
    policy = build_policy(policy_config, env_batch.envs[0].observation_space, env_batch.envs[0].action_space)
    return policy, env_batch


def test_evaluate_policy_per_task_single_task_returns_task_and_overall_summaries():
    env_config = load_env_config(
        "configs/env/multitask.yaml",
        override={
            "task_name": "goal_nav",
            "task_names": ["goal_nav"],
            "task_sampling_probs": {"goal_nav": 1.0, "coverage": 0.0, "formation": 0.0, "risk_nav": 0.0},
            "num_agents": 4,
            "max_steps": 2,
            "seed": 21,
        },
    )
    policy, env_batch = _build_mlp_policy_for_config(env_config)

    records, task_summaries, overall_summary = evaluate_policy_per_task(
        env_config,
        policy,
        ["goal_nav"],
        episodes_per_task=2,
        device=torch.device("cpu"),
        headless=True,
    )

    assert set(task_summaries) == {"goal_nav"}
    assert len(records) == 2
    assert all(record["task_name"] == "goal_nav" for record in records)
    assert task_summaries["goal_nav"]["task_name"] == "goal_nav"
    assert overall_summary["task_name"] == "overall"
    assert "return_mean" in overall_summary
    del env_batch


def test_evaluate_policy_per_task_multitask_separates_each_task_records():
    env_config = _base_multitask_env_config()
    policy, env_batch = _build_mlp_policy_for_config(env_config)

    records, task_summaries, overall_summary = evaluate_policy_per_task(
        env_config,
        policy,
        ["goal_nav", "coverage"],
        episodes_per_task=2,
        device=torch.device("cpu"),
        headless=True,
    )

    task_names = {record["task_name"] for record in records}
    assert task_names == {"goal_nav", "coverage"}
    assert len([record for record in records if record["task_name"] == "goal_nav"]) >= 2
    assert len([record for record in records if record["task_name"] == "coverage"]) >= 2
    assert set(task_summaries) == {"goal_nav", "coverage"}
    assert overall_summary["task_name"] == "overall"
    assert "return_mean" in overall_summary
    del env_batch


def test_make_fixed_task_eval_config_disables_random_multitask_sampling():
    base_env_config = _base_multitask_env_config()

    for task_name in ["goal_nav", "coverage"]:
        fixed_config = make_fixed_task_eval_config(base_env_config, task_name)
        assert fixed_config["task_name"] == task_name
        assert fixed_config["task_names"] == [task_name]
        assert fixed_config["task_sampling_probs"][task_name] == 1.0
        for other_task, probability in fixed_config["task_sampling_probs"].items():
            if other_task != task_name:
                assert probability == 0.0


def test_mock_final_eval_rows_have_per_task_plus_overall_rows():
    rows = []
    for num_agents in [4, 8]:
        env_config = load_env_config(
            "configs/env/multitask.yaml",
            override={
                "task_name": None,
                "task_names": ["goal_nav", "coverage"],
                "task_sampling_probs": {"goal_nav": 0.5, "coverage": 0.5, "formation": 0.0, "risk_nav": 0.0},
                "num_agents": num_agents,
                "max_steps": 2,
                "seed": 31,
            },
        )
        policy, env_batch = _build_mlp_policy_for_config(env_config)
        _, task_summaries, overall_summary = evaluate_policy_per_task(
            env_config,
            policy,
            ["goal_nav", "coverage"],
            episodes_per_task=1,
            device=torch.device("cpu"),
            headless=True,
        )
        rows.extend(
            [{"num_agents": num_agents, "task_name": task_name, **summary} for task_name, summary in task_summaries.items()]
        )
        rows.append({"num_agents": num_agents, "task_name": "overall", **overall_summary})
        del env_batch

    assert len(rows) == 2 * (2 + 1)


def test_ppo_periodic_eval_exposes_per_task_and_overall_metrics_with_compat_aliases():
    env_config = _base_multitask_env_config()
    train_config = load_generic_config("configs/policy/ppo_mlp.yaml")
    train_config.update(
        {
            "num_envs": 1,
            "rollout_steps": 2,
            "total_updates": 1,
            "target_episodes": 0,
            "epochs": 1,
            "minibatch_size": 1,
            "eval_interval": 1,
        }
    )
    env_batch = make_env_batch(env_config, int(train_config["num_envs"]))
    policy = build_policy(train_config, env_batch.envs[0].observation_space, env_batch.envs[0].action_space)
    trainer = PPOTrainer(env_batch, policy, train_config, device="cpu")
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "test_ppo_per_task_eval"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    history = trainer.train(
        output_dir,
        eval_env=env_batch.envs[0],
        eval_task_names=["goal_nav", "coverage"],
        eval_base_env_config=env_config,
        eval_episodes=1,
        headless=True,
    )

    record = history[-1]
    assert "eval_goal_nav_return" in record
    assert "eval_goal_nav_success_rate" in record
    assert "eval_coverage_return" in record
    assert "eval_coverage_success_rate" in record
    assert "eval_overall_return" in record
    assert "eval_overall_success_rate" in record
    assert "eval_reward" in record
    assert "eval_success_rate" in record
    assert float(record["eval_reward"]) == float(record["eval_overall_return"])
    assert float(record["eval_success_rate"]) == float(record["eval_overall_success_rate"])

    shutil.rmtree(output_dir)


def test_sac_periodic_eval_exposes_per_task_and_overall_metrics():
    env_config = _base_multitask_env_config()
    train_config = load_generic_config("configs/policy/sac_cnn_deepsets.yaml")
    train_config.update(
        {
            "num_envs": 1,
            "batch_size": 2,
            "warmup_steps": 0,
            "total_steps": 2,
            "eval_interval_steps": 1,
            "replay_size": 32,
        }
    )
    env_batch = make_env_batch(env_config, int(train_config["num_envs"]))
    policy = build_policy(train_config, env_batch.envs[0].observation_space, env_batch.envs[0].action_space)
    trainer = SACTrainer(env_batch, policy, train_config, device="cpu")
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "test_sac_per_task_eval"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    metrics = trainer.train(
        output_dir,
        eval_env=env_batch.envs[0],
        eval_task_names=["goal_nav", "coverage"],
        eval_base_env_config=env_config,
        eval_episodes=1,
        headless=True,
    )

    eval_records = [record for record in metrics if "eval_reward" in record]
    assert eval_records
    record = eval_records[0]
    assert "eval_goal_nav_return" in record
    assert "eval_coverage_return" in record
    assert "eval_overall_return" in record
    assert float(record["eval_reward"]) == float(record["eval_overall_return"])
    shutil.rmtree(output_dir)


def test_td3_periodic_eval_exposes_per_task_and_overall_metrics():
    env_config = _base_multitask_env_config()
    train_config = load_generic_config("configs/policy/td3_cnn_deepsets.yaml")
    train_config.update(
        {
            "num_envs": 1,
            "batch_size": 2,
            "warmup_steps": 0,
            "total_steps": 2,
            "eval_interval_steps": 1,
            "replay_size": 32,
        }
    )
    env_batch = make_env_batch(env_config, int(train_config["num_envs"]))
    policy = build_policy(train_config, env_batch.envs[0].observation_space, env_batch.envs[0].action_space)
    trainer = TD3Trainer(env_batch, policy, train_config, device="cpu")
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "test_td3_per_task_eval"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    metrics = trainer.train(
        output_dir,
        eval_env=env_batch.envs[0],
        eval_task_names=["goal_nav", "coverage"],
        eval_base_env_config=env_config,
        eval_episodes=1,
        headless=True,
    )

    eval_records = [record for record in metrics if "eval_reward" in record]
    assert eval_records
    record = eval_records[0]
    assert "eval_goal_nav_return" in record
    assert "eval_coverage_return" in record
    assert "eval_overall_return" in record
    assert float(record["eval_reward"]) == float(record["eval_overall_return"])
    shutil.rmtree(output_dir)


def test_evaluate_scaling_baseline_per_task_returns_task_breakdown_and_overall():
    env_config = _base_multitask_env_config()
    records, task_summaries, overall_summary = evaluate_baseline_per_task("random", env_config, ["goal_nav", "coverage"], episodes=1)

    assert records
    assert set(task_summaries) == {"goal_nav", "coverage"}
    assert overall_summary["task_name"] == "overall"
