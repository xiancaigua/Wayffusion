from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.linalg import eigh
from sklearn.cluster import KMeans


def symmetrize_affinity(affinity: np.ndarray, squash: bool = True) -> np.ndarray:
    matrix = np.asarray(affinity, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"affinity must be a square matrix, got shape={matrix.shape}")
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0.0)
    if squash:
        matrix = np.tanh(matrix)
    symmetric = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(symmetric, 0.0)
    return symmetric


def signed_spectral_embedding(
    affinity: np.ndarray,
    num_groups: int,
    tau_positive: float = 0.5,
    tau_negative: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(affinity, dtype=np.float64)
    if matrix.shape[0] < int(num_groups):
        raise ValueError("num_groups must be <= number of tasks")
    positive = np.where(matrix > 0.0, matrix, 0.0)
    negative = np.where(matrix < 0.0, -matrix, 0.0)
    d_positive = np.diag(positive.sum(axis=1))
    d_negative = np.diag(negative.sum(axis=1))
    l_positive = d_positive - positive
    l_negative = d_negative - negative
    lhs = l_positive + float(tau_negative) * d_negative
    rhs = l_negative + float(tau_positive) * d_positive
    # Small diagonal jitter keeps the generalized eigensystem well-conditioned
    # when a smoke run produces sparse or nearly zero affinities.
    rhs = rhs + np.eye(rhs.shape[0], dtype=np.float64) * 1e-8
    eigenvalues, eigenvectors = eigh(lhs, rhs)
    order = np.argsort(eigenvalues)
    selected = order[: int(num_groups)]
    return eigenvectors[:, selected], eigenvalues[selected]


def cluster_from_affinity(
    affinity: np.ndarray,
    task_names: list[str],
    num_groups: int,
    tau_positive: float = 0.5,
    tau_negative: float = 0.5,
    random_state: int = 42,
    n_init: int = 50,
) -> dict:
    symmetric = symmetrize_affinity(affinity, squash=True)
    embedding, eigenvalues = signed_spectral_embedding(
        symmetric,
        num_groups=num_groups,
        tau_positive=tau_positive,
        tau_negative=tau_negative,
    )
    labels = KMeans(n_clusters=int(num_groups), n_init=int(n_init), random_state=int(random_state)).fit_predict(embedding)
    grouped: dict[int, list[str]] = defaultdict(list)
    for task_name, label in zip(task_names, labels):
        grouped[int(label)].append(str(task_name))
    groups = [tasks for _, tasks in sorted(grouped.items())]
    return {
        "task_names": list(task_names),
        "num_groups": int(num_groups),
        "groups": groups,
        "labels": [int(label) for label in labels],
        "affinity": symmetric.tolist(),
        "embedding": embedding.tolist(),
        "eigenvalues": eigenvalues.tolist(),
    }


def write_grouping(path: str | Path, grouping: dict) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(grouping, handle, sort_keys=False, allow_unicode=True)
    return output_path


def load_grouping(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}

