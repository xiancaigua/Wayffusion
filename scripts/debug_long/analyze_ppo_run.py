from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _series(rows: list[dict[str, str]], key: str) -> list[float]:
    return [value for row in rows if (value := _float(row, key)) is not None]


def _summarize_group(rows: list[dict[str, str]], keys: list[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = _series(rows, key)
        if not values:
            continue
        summary[key] = {
            "mean": float(mean(values)),
            "min": float(min(values)),
            "max": float(max(values)),
            "last": float(values[-1]),
        }
    return summary


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--debug-dir", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _load_csv(run_dir / "training_metrics.csv")
    eval_rows = [row for row in train_rows if row.get("eval_success_rate", "") != ""]
    final_rows = _load_csv(run_dir / "eval_metrics.csv") if (run_dir / "eval_metrics.csv").exists() else []

    keys = [
        "eval_success_rate",
        "eval_reward",
        "eval_goal_nav_goal_coverage_ratio",
        "eval_collision_rate",
        "eval_path_length",
        "entropy",
        "mean_rollout_reward",
        "value_loss",
    ]
    first = eval_rows[:5]
    mid_start = max(len(eval_rows) // 2 - 2, 0)
    mid = eval_rows[mid_start : mid_start + 5]
    last = eval_rows[-5:]

    best_success_row = max(eval_rows, key=lambda row: _float(row, "eval_success_rate") or -1.0) if eval_rows else {}
    best_reward_row = max(eval_rows, key=lambda row: _float(row, "eval_reward") or -1e18) if eval_rows else {}
    final_overall = next((row for row in final_rows if row.get("task_name") == "overall"), final_rows[-1] if final_rows else {})

    payload = {
        "run_dir": str(run_dir),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "first5": _summarize_group(first, keys),
        "mid5": _summarize_group(mid, keys),
        "last5": _summarize_group(last, keys),
        "best_success_update": best_success_row.get("update"),
        "best_success_rate": _float(best_success_row, "eval_success_rate"),
        "best_reward_update": best_reward_row.get("update"),
        "best_eval_reward": _float(best_reward_row, "eval_reward"),
        "final_success_rate": _float(final_overall, "success_rate_mean"),
        "final_return": _float(final_overall, "return_mean"),
        "final_collision_rate": _float(final_overall, "collision_rate_mean"),
        "final_path_length": _float(final_overall, "path_length_mean"),
        "final_goal_coverage": _float(final_overall, "goal_coverage_ratio_mean"),
        "reference_random_return": _float(final_overall, "reference_random_return_mean"),
        "reference_heuristic_return": _float(final_overall, "reference_heuristic_return_mean"),
    }

    json_path = debug_dir / f"{args.label}_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        f"# PPO Run Summary: {args.label}",
        "",
        f"Run dir: `{run_dir}`",
        f"Training rows: {len(train_rows)}",
        f"Eval rows: {len(eval_rows)}",
        "",
        "## Headline",
        "",
        f"- Best eval success: {_fmt(payload['best_success_rate'])} at update {payload['best_success_update']}",
        f"- Best eval reward: {_fmt(payload['best_eval_reward'])} at update {payload['best_reward_update']}",
        f"- Final success: {_fmt(payload['final_success_rate'])}",
        f"- Final return: {_fmt(payload['final_return'])}",
        f"- Final goal coverage: {_fmt(payload['final_goal_coverage'])}",
        f"- Final collision rate: {_fmt(payload['final_collision_rate'])}",
        f"- Final path length: {_fmt(payload['final_path_length'])}",
        f"- Reference random return: {_fmt(payload['reference_random_return'])}",
        f"- Reference heuristic return: {_fmt(payload['reference_heuristic_return'])}",
        "",
        "## Eval Trend",
        "",
        "| segment | success | reward | goal coverage | collision | path | entropy | rollout reward | value loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, group in [("first5", payload["first5"]), ("mid5", payload["mid5"]), ("last5", payload["last5"])]:
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join(
                _fmt(group.get(key, {}).get("mean"))
                for key in [
                    "eval_success_rate",
                    "eval_reward",
                    "eval_goal_nav_goal_coverage_ratio",
                    "eval_collision_rate",
                    "eval_path_length",
                    "entropy",
                    "mean_rollout_reward",
                    "value_loss",
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## All Eval Points",
        "",
        "| update | success | reward | goal coverage | collision | path |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in eval_rows:
        lines.append(
            f"| {row.get('update')} | {_fmt(_float(row, 'eval_success_rate'))} | {_fmt(_float(row, 'eval_reward'))} | "
            f"{_fmt(_float(row, 'eval_goal_nav_goal_coverage_ratio'))} | {_fmt(_float(row, 'eval_collision_rate'))} | "
            f"{_fmt(_float(row, 'eval_path_length'))} |"
        )

    md_path = debug_dir / f"{args.label}_summary.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"summary_md={md_path}")
    print(f"summary_json={json_path}")


if __name__ == "__main__":
    main()
