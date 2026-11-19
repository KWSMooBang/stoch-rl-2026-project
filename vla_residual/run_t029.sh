#!/usr/bin/env bash
# T029 success-hold: T022 PPO / GRPO (configs/t022_{ppo,grpo}.yaml, identical alpha / obs / hyper-params / env steps /
# eval ids) with the TRAIN reward switched to "hold" (H1): r_t = h_t - h_{t-1} (h_t = success at the end of decision t),
# no success termination, truncation at decision 16 -> episode return = success_at_end (= eval protocol).
# usage: vla_residual/run_t029.sh <ppo|grpo> <seed> [extra hydra overrides]
set -uo pipefail
ALGO=${1:?algo}; SEED=${2:?seed}; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
case $ALGO in ppo|grpo) ;; *) echo "T029 runs only ppo|grpo, got $ALGO" >&2; exit 2 ;; esac
# human instruction (T029): physical GPU 0 ONLY (GPU 1 and 3 forbidden). Override list via T029_GPUS.
case " ${T029_GPUS:-0} " in
  *" ${CUDA_VISIBLE_DEVICES:-none} "*) ;;
  *) echo "[t029] REFUSE: CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}' (allowed: ${T029_GPUS:-0})"; exit 3 ;;
esac
export RUN_ID=${RUN_ID:-t029-$ALGO-hold-s$SEED}
export RUN_DIR=${RUN_DIR:-/workspace/stochrl_t029/scratch/$RUN_ID}
exec "$HERE/run.sh" "$ALGO" "$SEED" +env.train.vla_residual.reward=hold "$@"
