from __future__ import annotations


def common_reward(config: dict, transition_info: dict) -> tuple[float, dict[str, float]]:
    weights = config["reward_weights"]["common"]
    num_agents = max(int(transition_info.get("num_agents", 1)), 1)
    spatial_scale = max(float(transition_info.get("spatial_scale", 1.0)), 1.0)
    max_step_distance = max(float(transition_info.get("max_step_distance", 1.0)), 1e-6)
    collision_rate = float(transition_info["pair_collision_count"]) / float(num_agents)
    obstacle_rate = float(transition_info["obstacle_collision_count"]) / float(num_agents)
    path_effort = float(transition_info["path_length_delta"]) / float(num_agents * max_step_distance)
    safety_rate = float(transition_info["step_safety_violations"]) / float(num_agents)
    risk_rate = float(transition_info["step_risk_exposure"]) / float(num_agents)
    components = {
        "collision_penalty": float(weights["collision"] * collision_rate),
        "obstacle_penalty": float(weights["obstacle_collision"] * obstacle_rate),
        "path_penalty": float(weights["path_length"] * path_effort),
        "time_penalty": float(weights["time"] / spatial_scale),
        "safety_penalty": float(weights["safety_violation"] * safety_rate),
        "risk_penalty": float(weights["risk"] * risk_rate),
    }
    total = float(
        components["collision_penalty"]
        + components["obstacle_penalty"]
        + components["path_penalty"]
        + components["time_penalty"]
        + components["safety_penalty"]
        + components["risk_penalty"]
    )
    return total, components
