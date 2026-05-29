from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _series(rows: list[dict[str, str]], x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x = _float(row.get(x_key))
        y = _float(row.get(y_key))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def _save_line_plot(rows: list[dict[str, str]], x_key: str, y_key: str, output_path: Path) -> bool:
    xs, ys = _series(rows, x_key, y_key)
    if not xs:
        return False
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, marker="o", linewidth=2)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(f"{y_key} vs {x_key}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def _save_eval_bar(rows: list[dict[str, str]], output_path: Path) -> bool:
    if not rows:
        return False
    target = None
    for row in rows:
        if row.get("task_name") == "overall":
            target = row
            break
    if target is None:
        target = rows[-1]
    candidates = [
        ("return_mean", _float(target.get("return_mean"))),
        ("success_rate_mean", _float(target.get("success_rate_mean"))),
        ("collision_rate_mean", _float(target.get("collision_rate_mean"))),
        ("goal_coverage_ratio_mean", _float(target.get("goal_coverage_ratio_mean"))),
        ("coverage_ratio_mean", _float(target.get("coverage_ratio_mean"))),
        ("formation_error_mean", _float(target.get("formation_error_mean"))),
        ("normalized_score_mean", _float(target.get("normalized_score_mean"))),
    ]
    labels = [label for label, value in candidates if value is not None]
    values = [value for _label, value in candidates if value is not None]
    if not values:
        return False
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values)
    ax.set_title("Final Eval Summary")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--x-key", default=None)
    parser.add_argument("--metrics", nargs="+", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    plots_dir = run_dir / "plots"
    training_path = run_dir / "training_metrics.csv"
    eval_path = run_dir / "eval_metrics.csv"
    rows = _load_csv(training_path) if training_path.exists() else []
    x_key = args.x_key
    if x_key is None:
        for candidate in ("update", "epoch", "step"):
            if rows and candidate in rows[0]:
                x_key = candidate
                break
    if x_key is not None:
        for metric in args.metrics:
            _save_line_plot(rows, x_key, metric, plots_dir / f"{metric}.png")
    if eval_path.exists():
        _save_eval_bar(_load_csv(eval_path), plots_dir / "final_eval_summary.png")


if __name__ == "__main__":
    main()
