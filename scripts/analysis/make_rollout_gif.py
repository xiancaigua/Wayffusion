from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-path", default="outputs/rollout.gif")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    images = []
    for path in sorted(input_dir.glob("*.png")):
        images.append(imageio.imread(path))
    if not images:
        raise FileNotFoundError(f"No PNG files found in {input_dir}")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, images, duration=0.8)


if __name__ == "__main__":
    main()
