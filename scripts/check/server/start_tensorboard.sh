#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Docker should be started with --network=host. In that mode, container
# 127.0.0.1:6006 is the server's 127.0.0.1:6006; access it from a local
# machine with: ssh -N -L 16006:127.0.0.1:6006 user@server
tensorboard --logdir outputs/training --host 127.0.0.1 --port 6006
