from __future__ import annotations

import time

import psutil
import torch


def get_memory_usage_mb() -> float:
    process = psutil.Process()
    rss_mb = process.memory_info().rss / (1024.0 * 1024.0)
    if torch.cuda.is_available():
        rss_mb += torch.cuda.memory_allocated() / (1024.0 * 1024.0)
    return float(rss_mb)


def measure_policy_latency_ms(policy, obs_tensors: dict[str, torch.Tensor], repeats: int = 20) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            if hasattr(policy, "act_deterministic"):
                policy.act_deterministic(obs_tensors)
            else:
                policy(obs_tensors)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return float(1000.0 * elapsed / max(repeats, 1))
