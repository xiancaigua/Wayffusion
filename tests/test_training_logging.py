import shutil
from pathlib import Path

from scripts._common import build_metric_logger, tensorboard_dir


def test_tensorboard_logger_writes_event_file_and_console_output(capsys):
    run_dir = Path(__file__).resolve().parents[1] / "outputs" / "test_artifacts" / "tb_logging_run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    writer, log_record = build_metric_logger(
        run_dir,
        namespace="ppo/train",
        step_key="update",
        tensorboard_enabled=True,
        console_interval=1,
        key_order=["mean_rollout_reward", "policy_loss", "eval_reward"],
    )

    assert writer is not None
    log_record({"update": 1, "mean_rollout_reward": 1.5, "policy_loss": 0.25})
    log_record({"update": 2, "eval_reward": 2.0, "eval_success_rate": 0.5, "checkpoint_path": "dummy.pt"})
    writer.close()

    tb_dir = tensorboard_dir(run_dir)
    assert any(path.name.startswith("events.out.tfevents") for path in tb_dir.iterdir())

    stdout = capsys.readouterr().out
    assert "update=1" in stdout
    assert "eval_reward=2.000" in stdout

    shutil.rmtree(run_dir)
