import torch
from torch import nn

from algorithms.bc import BCTrainer


def test_bc_permutation_waypoint_loss_matches_swapped_targets():
    trainer = BCTrainer(
        nn.Linear(1, 1),
        {
            "learning_rate": 1e-3,
            "bc_waypoint_step": 0.5,
            "permutation_loss_max_agents": 4,
        },
        device="cpu",
    )
    agents = torch.zeros((1, 2, 6), dtype=torch.float32)
    agents[0, 1, 0] = 1.0
    prediction = torch.tensor([[[0.0, 2.0], [0.0, 2.0]]], dtype=torch.float32)
    swapped_target = torch.tensor([[[2.0, 2.0], [-2.0, 2.0]]], dtype=torch.float32)
    mask = torch.ones((1, 2), dtype=torch.bool)

    loss = trainer._permutation_waypoint_mse(prediction, swapped_target, mask, agents)

    assert float(loss.item()) == 0.0
