from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--x-key", default="episode")
    parser.add_argument("--y-key", default="normalized_score")
    parser.add_argument("--output-path", default="outputs/metrics_plot.png")
    args = parser.parse_args()

    rows = []
    with open(args.csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    x_vals = [float(row[args.x_key]) for row in rows if row.get(args.x_key)]
    y_vals = [float(row[args.y_key]) for row in rows if row.get(args.y_key)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_vals, y_vals, marker="o")
    ax.set_xlabel(args.x_key)
    ax.set_ylabel(args.y_key)
    ax.set_title(f"{args.y_key} vs {args.x_key}")
    fig.tight_layout()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
