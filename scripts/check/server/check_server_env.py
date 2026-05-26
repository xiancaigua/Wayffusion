from __future__ import annotations

import importlib.util
import os
import platform
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    root = _repo_root()
    print(f"python={sys.version.split()[0]}")
    print(f"python_executable={sys.executable}")
    print(f"platform={platform.platform()}")

    try:
        import torch

        print(f"torch={torch.__version__}")
        print(f"cuda_available={torch.cuda.is_available()}")
        print(f"torch_cuda={torch.version.cuda}")
        print(f"gpu_count={torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            print(f"gpu_{idx}={torch.cuda.get_device_name(idx)}")
    except Exception as exc:
        print(f"torch_error={exc}")

    print(f"MPLBACKEND={os.environ.get('MPLBACKEND', '')}")
    print(f"http_proxy={os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY') or ''}")
    print(f"https_proxy={os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY') or ''}")
    print(f"psutil_importable={_module_available('psutil')}")
    print(f"tensorboard_importable={_module_available('tensorboard')}")
    print(f"cwd={Path.cwd()}")
    print(f"repo_root={root}")

    required_paths = ["README.md", "configs", "scripts"]
    missing = [name for name in required_paths if not (root / name).exists()]
    if missing:
        print(f"repo_layout_ok=False missing={','.join(missing)}")
        return 1
    print("repo_layout_ok=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
