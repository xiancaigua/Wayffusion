from __future__ import annotations

import torch
from torch.distributions import Normal


class SquashedNormal:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        self.mean = mean
        self.std = std
        self.eps = float(eps)
        self.base_dist = Normal(mean, std)

    def _clamp_action(self, action: torch.Tensor) -> torch.Tensor:
        return torch.clamp(action, -1.0 + self.eps, 1.0 - self.eps)

    def atanh(self, action: torch.Tensor) -> torch.Tensor:
        bounded = self._clamp_action(action)
        return 0.5 * (torch.log1p(bounded) - torch.log1p(-bounded))

    def _log_prob_from_raw(self, raw_action: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        bounded = self._clamp_action(action)
        correction = torch.log(torch.clamp(1.0 - bounded.pow(2), min=self.eps))
        return self.base_dist.log_prob(raw_action) - correction

    def sample(self, return_raw: bool = False):
        raw_action = self.base_dist.sample()
        action = torch.tanh(raw_action)
        if return_raw:
            return action, raw_action
        return action

    def rsample(self, return_raw: bool = False):
        raw_action = self.base_dist.rsample()
        action = torch.tanh(raw_action)
        if return_raw:
            return action, raw_action
        return action

    def deterministic(self) -> torch.Tensor:
        return torch.tanh(self.mean)

    def log_prob(
        self,
        action: torch.Tensor,
        raw_action: torch.Tensor | None = None,
        reduce: bool = True,
    ) -> torch.Tensor:
        bounded_action = self._clamp_action(action)
        raw = raw_action if raw_action is not None else self.atanh(bounded_action)
        log_prob = self._log_prob_from_raw(raw, bounded_action)
        if not reduce:
            return log_prob
        reduce_dims = tuple(range(1, log_prob.ndim))
        return log_prob.sum(dim=reduce_dims) if reduce_dims else log_prob

    def entropy_estimate(
        self,
        action: torch.Tensor,
        raw_action: torch.Tensor | None = None,
        reduce: bool = True,
    ) -> torch.Tensor:
        return -self.log_prob(action, raw_action=raw_action, reduce=reduce)
