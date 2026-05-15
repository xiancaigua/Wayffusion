import shutil
from pathlib import Path

from algorithms import PPOTrainer
from policies import build_policy
from scripts._common import load_env_config, load_generic_config
from utils import make_env_batch


def test_ppo_target_episodes_stops_early():
    env_config = load_env_config(
        "configs/env/multitask.yaml",
        override={"task_name": "goal_nav", "max_steps": 1, "num_agents": 4},
    )
    train_config = load_generic_config("configs/policy/ppo_mlp.yaml")
    train_config.update(
        {
            "num_envs": 1,
            "rollout_steps": 2,
            "total_updates": 10,
            "target_episodes": 3,
            "eval_interval": 100,
        }
    )
    env_batch = make_env_batch(env_config, int(train_config["num_envs"]))
    policy = build_policy(train_config, env_batch.envs[0].observation_space, env_batch.envs[0].action_space)
    trainer = PPOTrainer(env_batch, policy, train_config, device="cpu")
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "test_ppo_episode_budget"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    history = trainer.train(output_dir, eval_env=env_batch.envs[0], eval_episodes=1, headless=True)

    assert trainer.completed_episodes >= 3
    assert len(history) < int(train_config["total_updates"])
    assert int(history[-1]["cumulative_episodes"]) >= 3
    shutil.rmtree(output_dir)
