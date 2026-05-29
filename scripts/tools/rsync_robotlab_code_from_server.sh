#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE="ycl@10.12.120.237:/srv/shared/home/ycl/workspace/robotlab_train/"
LOCAL_HOSTNAME="$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown)"

if [[ "$LOCAL_HOSTNAME" == "venus" ]]; then
  echo "Error: rsync_robotlab_code_from_server.sh must be run on your local machine, not on server '$LOCAL_HOSTNAME'." >&2
  echo "Run it in your local repo checkout to pull code from 10.12.120.237." >&2
  exit 1
fi

cd "$REPO_ROOT"

rsync -avz \
  -e "ssh -p 11222" \
  --exclude='.git' \
  --exclude='.vscode' \
  --exclude='.idea' \
  --exclude='.thumbs' \
  --exclude='.pytest_cache' \
  --exclude='__pycache__' \
  --exclude='logs' \
  --exclude='runs' \
  --exclude='recordings' \
  --exclude='output' \
  --exclude='outputs' \
  --exclude='videos' \
  --exclude='wandb' \
  --exclude='.neptune' \
  --exclude='*.dmp' \
  --exclude='.DS_Store' \
  --exclude='*.egg-info' \
  --exclude='*.pyc' \
  --exclude='*.pb' \
  --exclude='*.pt' \
  --exclude='*.tmp' \
  --exclude='*.npz' \
  --exclude='_isaac_sim*' \
  --exclude='_repo' \
  --exclude='_build' \
  --exclude='.lastformat' \
  --exclude='pyrightconfig.json' \
  --exclude='docker/artifacts' \
  "$SOURCE" ./
