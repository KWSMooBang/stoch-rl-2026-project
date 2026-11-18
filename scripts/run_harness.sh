#!/usr/bin/env bash
# Harness entry for T014. Training runs on CPU only (a process pool); the GPU the harness hands
# out is not used. Refuses GPU 3 (forbidden on this server).
# usage (from the worktree root):
#   RUN_HOURS=1 /harness/launch.sh t014-p1-baselines 1 bash scripts/run_harness.sh --phase p1 --workers 40
set -euo pipefail
case ",${CUDA_VISIBLE_DEVICES:-}," in
  *,3,*) echo "refusing: harness assigned GPU 3 (forbidden)" >&2; exit 3 ;;
esac
echo "assigned GPU(s): ${CUDA_VISIBLE_DEVICES:-none} (unused: CPU-only run)"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$(dirname "$0")/.."
exec uv run --frozen python -m aapr.run_phase --out "$RUN_DIR" "$@"
