from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import save_expert_dataset


def _atomic_save_dataset(path: Path, payload: dict[str, np.ndarray]) -> None:
    tmp_path = path.with_name(path.stem + ".tmp" + path.suffix)
    save_expert_dataset(tmp_path, payload)
    os.replace(tmp_path, path)


def _atomic_write_text(path: Path, content: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_paths = [Path(path) for path in args.inputs]
    datasets = [np.load(path, allow_pickle=True) for path in input_paths]
    keys = list(datasets[0].files)
    for path, dataset in zip(input_paths, datasets):
        missing = set(keys) - set(dataset.files)
        extra = set(dataset.files) - set(keys)
        if missing or extra:
            raise ValueError(f"Dataset key mismatch for {path}: missing={sorted(missing)} extra={sorted(extra)}")

    payload = {}
    for key in keys:
        payload[key] = np.concatenate([dataset[key] for dataset in datasets], axis=0)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save_dataset(output_path, payload)

    summary = {
        "output": str(output_path),
        "inputs": [str(path) for path in input_paths],
        "input_samples": [int(dataset[keys[0]].shape[0]) for dataset in datasets],
        "samples": int(payload[keys[0]].shape[0]),
    }
    _atomic_write_text(output_path.with_suffix(".summary.json"), json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"saved_dataset={output_path}")
    print(f"summary={output_path.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
