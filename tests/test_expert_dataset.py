from pathlib import Path

import numpy as np

from utils import ExpertDataset, load_expert_dataset, save_expert_dataset


def test_expert_dataset_round_trip():
    dataset_path = Path("outputs/test_tmp/expert_unit_test.npz")
    payload = {
        "task_field": np.zeros((2, 9, 8, 8), dtype=np.float32),
        "agents": np.zeros((2, 6, 6), dtype=np.float32),
        "task_id": np.zeros((2, 4), dtype=np.float32),
        "global_info": np.ones((2, 5), dtype=np.float32),
        "agent_mask": np.array([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=bool),
        "action": np.zeros((2, 6, 2), dtype=np.float32),
        "num_agents": np.array([4, 2], dtype=np.int64),
        "reward": np.array([1.0, 0.0], dtype=np.float32),
        "done": np.array([0.0, 1.0], dtype=np.float32),
    }
    save_expert_dataset(dataset_path, payload)

    loaded = load_expert_dataset(dataset_path)
    assert loaded["task_field"].shape == (2, 9, 8, 8)
    assert loaded["agent_mask"].dtype == np.bool_

    dataset = ExpertDataset(dataset_path)
    sample = dataset[0]
    assert len(dataset) == 2
    assert sample["agents"].shape == (6, 6)
    assert sample["action"].shape == (6, 2)
    assert int(sample["num_agents"].item()) == 4
