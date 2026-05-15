from __future__ import annotations

from baselines.geometric_formation import GeometricFormationPolicy
from baselines.greedy_coverage import GreedyCoveragePolicy
from baselines.greedy_goal import GreedyGoalPolicy
from baselines.random_policy import RandomWaypointPolicy
from baselines.risk_potential import RiskAwarePotentialPolicy
from tasks import TASK_ORDER


class HeuristicPolicy:
    def __init__(self, config: dict):
        self.task_to_policy = {
            "goal_nav": GreedyGoalPolicy(config),
            "coverage": GreedyCoveragePolicy(config),
            "formation": GeometricFormationPolicy(config),
            "risk_nav": RiskAwarePotentialPolicy(config),
        }

    def act(self, observation: dict):
        task_idx = int(observation["task_id"].argmax())
        task_name = TASK_ORDER[task_idx]
        return self.task_to_policy[task_name].act(observation)


def make_baseline(policy_name: str, config: dict):
    if policy_name == "random":
        return RandomWaypointPolicy(config)
    if policy_name == "heuristic":
        return HeuristicPolicy(config)
    if policy_name == "greedy_goal":
        return GreedyGoalPolicy(config)
    if policy_name == "greedy_coverage":
        return GreedyCoveragePolicy(config)
    if policy_name == "geometric_formation":
        return GeometricFormationPolicy(config)
    if policy_name == "risk_potential":
        return RiskAwarePotentialPolicy(config)
    raise ValueError(f"Unknown baseline policy: {policy_name}")


__all__ = [
    "RandomWaypointPolicy",
    "GreedyGoalPolicy",
    "GreedyCoveragePolicy",
    "GeometricFormationPolicy",
    "RiskAwarePotentialPolicy",
    "HeuristicPolicy",
    "make_baseline",
]
