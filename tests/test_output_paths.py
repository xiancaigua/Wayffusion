import shutil
from pathlib import Path

import torch

from scripts._common import latest_checkpoint, run_dir_from_checkpoint, save_run_snapshot, tensorboard_dir, timestamped_training_dir


def test_timestamped_training_dir_inserts_time_layer():
    output_dir = timestamped_training_dir("ppo", "demo_run", timestamp="20260514_120000")

    assert output_dir.name == "demo_run"
    assert output_dir.parent.name == "20260514_120000"
    assert output_dir.parent.parent.name == "ppo"


def test_snapshot_and_recursive_checkpoint_lookup():
    run_dir = Path(__file__).resolve().parents[1] / "outputs" / "test_artifacts" / "demo_run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    save_run_snapshot(
        run_dir,
        train_config={"name": "demo"},
        env_config={"task_name": "goal_nav"},
        cli_args={"headless": True},
        model_state_dict={"weight": torch.zeros(1)},
        extra_metadata={"task_names": ["goal_nav"]},
    )
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint_0001.pt"
    torch.save({"model_state_dict": {"weight": torch.ones(1)}}, checkpoint_path)

    assert (run_dir / "snapshot" / "train_config.yaml").exists()
    assert (run_dir / "snapshot" / "env_config.yaml").exists()
    assert (run_dir / "snapshot" / "cli_args.yaml").exists()
    assert (run_dir / "snapshot" / "metadata.yaml").exists()
    assert (run_dir / "snapshot" / "initial_model.pt").exists()
    assert latest_checkpoint(run_dir) == checkpoint_path
    assert run_dir_from_checkpoint(checkpoint_path) == run_dir

    shutil.rmtree(run_dir)


def test_tensorboard_dir_is_sibling_under_run_dir():
    output_dir = timestamped_training_dir("ppo", "tb_demo", timestamp="20260515_120000")
    tb_dir = tensorboard_dir(output_dir)

    assert tb_dir.name == "tensorboard"
    assert tb_dir.parent == output_dir

    shutil.rmtree(output_dir.parent)
