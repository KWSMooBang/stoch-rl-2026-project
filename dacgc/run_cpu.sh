#!/usr/bin/env bash
# Harness wrapper: refuse GPU 3, then run CPU-only (the harness still reserves one GPU).
set -euo pipefail
echo "assigned CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
if [[ ",${CUDA_VISIBLE_DEVICES:-}," == *",3,"* ]]; then
  echo "GPU 3 assigned -> refusing (T013 forbids GPU 3)" >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
start=$(date +%s)
uv run python -m dacgc.harness_main "$@"
echo "wall_clock_s=$(( $(date +%s) - start ))"
