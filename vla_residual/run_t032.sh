#!/usr/bin/env bash
# T032: E0 = GRPO residual (same config as t022-grpo-s0 / t030-grpo-s1,s2 = E2) with the VLM feature block of the
# observation set to constant 0. One flag (vla_residual.zero_vlm) set for both env splits; everything else
# (obs dim 2092, MLP, alpha, reward, lr, env steps, eval seed 1000, held-out 128) is unchanged.
#   e0 -> run.sh grpo <seed> env.{train,eval}.vla_residual.zero_vlm=True
#   e2 -> run.sh grpo <seed>                       (flag at its default False; not used for T032 runs)
# usage: vla_residual/run_t032.sh <e0|e2> <seed> [extra hydra overrides (smoke only)]
set -uo pipefail
COND=${1:?cond}; SEED=${2:?seed}; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
# human instruction (2026-11-20): T032 runs on physical GPU 0 ONLY (GPU 1 and 3 forbidden).
if [ "${CUDA_VISIBLE_DEVICES:-none}" != 0 ]; then
  echo "[t032] REFUSE: CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}' (allowed: 0)"; exit 3
fi
export RUN_ID=${RUN_ID:-t032-grpo-$COND-s$SEED}
export RUN_DIR=${RUN_DIR:-/workspace/stochrl_t032/scratch/$RUN_ID}
case $COND in
  e0) exec "$HERE/run.sh" grpo "$SEED" env.train.vla_residual.zero_vlm=True env.eval.vla_residual.zero_vlm=True "$@" ;;
  e2) exec "$HERE/run.sh" grpo "$SEED" "$@" ;;
  *) echo "[t032] unknown cond $COND" >&2; exit 2 ;;
esac
