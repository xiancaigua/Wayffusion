import shutil
from pathlib import Path

import torch

from envs import CentralizedMultiUAVEnv
from policies import build_policy
from scripts._common import load_env_config, load_generic_config
from utils import evaluate_policy_episodes


def _build_smoke_policy_and_env():
    env_config = load_env_config(
        "configs/env/multitask.yaml",
        override={"task_name": "goal_nav", "max_steps": 4, "num_agents": 4},
    )
    env = CentralizedMultiUAVEnv(env_config)
    policy_config = load_generic_config("configs/policy/ppo_mlp.yaml")
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    return env, policy


def test_evaluate_policy_records_gif():
    env, policy = _build_smoke_policy_and_env()
    media_dir = Path(__file__).resolve().parents[1] / "outputs" / "test_media"
    if media_dir.exists():
        shutil.rmtree(media_dir)
    records = evaluate_policy_episodes(
        env,
        policy,
        episodes=1,
        device=torch.device("cpu"),
        headless=True,
        record_dir=media_dir,
        record_episodes=1,
        record_format="gif",
        record_fps=4,
        record_prefix="smoke",
    )

    assert len(records) == 1
    assert "recording_path" in records[0]
    recording_path = Path(records[0]["recording_path"])
    assert recording_path.exists()
    assert recording_path.suffix == ".gif"
    shutil.rmtree(media_dir)


def test_evaluate_policy_uses_human_render_when_not_headless():
    env, policy = _build_smoke_policy_and_env()
    render_modes = []
    original_render = env.render

    def wrapped_render(mode: str = "human"):
        render_modes.append(mode)
        if mode == "human":
            return None
        return original_render(mode)

    env.render = wrapped_render
    evaluate_policy_episodes(
        env,
        policy,
        episodes=1,
        device=torch.device("cpu"),
        headless=False,
        record_episodes=0,
    )

    assert "human" in render_modes
