#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE="ycl@10.12.120.237:/srv/shared/home/ycl/workspace/robotlab_train/"
LOCAL_HOSTNAME="$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown)"

if [[ "$LOCAL_HOSTNAME" == "venus" ]]; then
  echo "Error: rsync_robotlab_results_from_server.sh must be run on your local machine, not on server '$LOCAL_HOSTNAME'." >&2
  echo "Run it in your local repo checkout to pull results back from 10.12.120.237." >&2
  exit 1
fi

cd "$REPO_ROOT"

rsync -avz \
  -e "ssh -p 11222" \
  --prune-empty-dirs \
  --include='*/' \
  --include='/*.md' \
  --exclude='logs/*.out' \
  --exclude='logs/*.err' \
  --exclude='**/logs/*.out' \
  --exclude='**/logs/*.err' \
  --include='logs/***' \
  --include='**/logs/***' \
  --include='runs/***' \
  --include='**/runs/***' \
  --include='recordings/***' \
  --include='**/recordings/***' \
  --include='output/***' \
  --include='**/output/***' \
  --include='outputs/***' \
  --include='**/outputs/***' \
  --include='videos/***' \
  --include='**/videos/***' \
  --include='wandb/***' \
  --include='**/wandb/***' \
  --include='.neptune/***' \
  --include='**/.neptune/***' \
  --exclude='*' \
  "$SOURCE" ./
