from __future__ import annotations

import numpy as np


class RandomWaypointPolicy:
    def __init__(self, config: dict, seed: int = 0):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def act(self, observation: dict) -> np.ndarray:
        num_agents = observation["agents"].shape[0]
        return self.rng.uniform(-1.0, 1.0, size=(num_agents, 2)).astype(np.float32)
