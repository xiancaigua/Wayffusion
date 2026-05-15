import numpy as np

from policies import build_policy, observation_to_tensor
from scripts._common import load_env_config, load_generic_config
from envs import CentralizedMultiUAVEnv


def test_variable_agent_policies_output_joint_waypoints():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 10, "task_name": "goal_nav"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=5)

    for policy_path in [
        "configs/policy/ppo_cnn_deepsets.yaml",
        "configs/policy/ppo_attention.yaml",
    ]:
        policy_config = load_generic_config(policy_path)
        policy = build_policy(policy_config, env.observation_space, env.action_space)
        obs_tensor = observation_to_tensor(observation, device="cpu")
        action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
        assert action.shape == (1, config["num_agents"], 2)
        assert np.all(np.isfinite(action))


def test_mlp_policy_stays_fixed_n():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": "coverage"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=7)
    policy_config = load_generic_config("configs/policy/ppo_mlp.yaml")
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
    assert action.shape == (1, 4, 2)
