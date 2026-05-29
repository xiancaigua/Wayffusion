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


def test_cnn_deepsets_policy_supports_optional_coordination_repulsion():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": "coverage"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=9)
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["coordination_repulsion_strength"] = 0.25
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
    assert action.shape == (1, 4, 2)
    assert np.all(np.isfinite(action))


def test_cnn_deepsets_policy_supports_optional_spatial_action_head():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": "coverage"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=11)
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["use_spatial_action_head"] = True
    policy_config["spatial_action_strength"] = 0.5
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
    assert action.shape == (1, 4, 2)
    assert np.all(np.isfinite(action))


def test_cnn_deepsets_policy_supports_spatial_target_suppression():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": "coverage"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=13)
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["use_spatial_action_head"] = True
    policy_config["spatial_action_strength"] = 0.5
    policy_config["spatial_target_suppression_strength"] = 1.0
    policy_config["coordination_repulsion_strength"] = 0.25
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
    assert action.shape == (1, 4, 2)
    assert np.all(np.isfinite(action))


def test_cnn_deepsets_policy_supports_angular_slot_embeddings():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": "coverage"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=15)
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["use_spatial_action_head"] = True
    policy_config["spatial_action_strength"] = 0.5
    policy_config["coordination_repulsion_strength"] = 0.25
    policy_config["use_angular_slot_embeddings"] = True
    policy_config["slot_embedding_strength"] = 1.0
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
    assert action.shape == (1, 4, 2)
    assert np.all(np.isfinite(action))


def test_cnn_deepsets_policy_supports_sector_target_bias():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": "coverage"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=17)
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["use_spatial_action_head"] = True
    policy_config["spatial_action_strength"] = 0.5
    policy_config["coordination_repulsion_strength"] = 0.25
    policy_config["use_angular_slot_embeddings"] = True
    policy_config["slot_embedding_strength"] = 1.0
    policy_config["sector_target_bias_strength"] = 0.35
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
    assert action.shape == (1, 4, 2)
    assert np.all(np.isfinite(action))


def test_cnn_deepsets_policy_supports_global_slot_head():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": "coverage"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=19)
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["use_global_slot_head"] = True
    policy_config["global_slot_strength"] = 0.5
    policy_config["use_angular_slot_embeddings"] = True
    policy_config["slot_embedding_strength"] = 1.0
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
    assert action.shape == (1, 4, 2)
    assert np.all(np.isfinite(action))


def test_cnn_deepsets_global_slot_head_is_compatible_across_tasks():
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["use_global_slot_head"] = True
    policy_config["global_slot_strength"] = 0.4
    policy_config["use_angular_slot_embeddings"] = True
    policy_config["slot_embedding_strength"] = 1.0
    for task_name in ["goal_nav", "coverage", "formation", "risk_nav"]:
        config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": task_name})
        env = CentralizedMultiUAVEnv(config)
        observation, _ = env.reset(seed=23)
        policy = build_policy(policy_config, env.observation_space, env.action_space)
        obs_tensor = observation_to_tensor(observation, device="cpu")
        action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
        assert action.shape == (1, 4, 2)
        assert np.all(np.isfinite(action))


def test_cnn_deepsets_policy_supports_slot_dominant_actor():
    config = load_env_config("configs/env/multitask.yaml", override={"num_agents": 4, "task_name": "coverage"})
    env = CentralizedMultiUAVEnv(config)
    observation, _ = env.reset(seed=29)
    policy_config = load_generic_config("configs/policy/ppo_cnn_deepsets.yaml")
    policy_config["use_global_slot_head"] = True
    policy_config["global_slot_strength"] = 0.8
    policy_config["use_angular_slot_embeddings"] = True
    policy_config["slot_embedding_strength"] = 1.0
    policy_config["actor_mean_residual_weight"] = 0.0
    policy = build_policy(policy_config, env.observation_space, env.action_space)
    obs_tensor = observation_to_tensor(observation, device="cpu")
    action = policy.act_deterministic(obs_tensor).detach().cpu().numpy()
    assert action.shape == (1, 4, 2)
    assert np.all(np.isfinite(action))
