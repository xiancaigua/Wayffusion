import numpy as np

from baselines import make_baseline
from scripts._common import load_env_config
from envs import CentralizedMultiUAVEnv


def test_baselines_output_valid_actions():
    config = load_env_config("configs/env/multitask.yaml")
    env = CentralizedMultiUAVEnv({**config, "task_name": "goal_nav"})
    obs, _ = env.reset(seed=17)
    for name in ["random", "greedy_goal", "greedy_coverage", "geometric_formation", "risk_potential", "heuristic"]:
        action = make_baseline(name, config).act(obs)
        assert action.shape == env.action_space.shape
        assert np.all(action <= 1.0 + 1e-6)
        assert np.all(action >= -1.0 - 1e-6)


def test_heuristic_beats_random_on_goal_nav_smoke():
    config = load_env_config("configs/env/multitask.yaml", override={"task_name": "goal_nav"})
    env = CentralizedMultiUAVEnv(config)
    heuristic = make_baseline("heuristic", config)
    random_policy = make_baseline("random", config)

    heuristic_scores = []
    random_scores = []
    for seed in range(5):
        _, _ = env.reset(seed=100 + seed)
        done = False
        info = {}
        while not done:
            obs = env.last_observation
            obs, _, terminated, truncated, info = env.step(heuristic.act(obs))
            done = terminated or truncated
        heuristic_scores.append(info["normalized_score"])

        _, _ = env.reset(seed=100 + seed)
        done = False
        while not done:
            obs = env.last_observation
            obs, _, terminated, truncated, info = env.step(random_policy.act(obs))
            done = terminated or truncated
        random_scores.append(info["normalized_score"])

    assert float(np.mean(heuristic_scores)) > float(np.mean(random_scores))
