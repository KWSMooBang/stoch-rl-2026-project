#!/usr/bin/env bash
# T030: seed replication of the main residual RL results. No code change: delegates to the existing run tools
#   grpo    -> run.sh grpo <seed>                (T022 sparse GRPO, configs/t022_grpo.yaml; same as t022-grpo-s0)
#   ppo     -> run.sh ppo <seed>                 (T022 PPO beta=0, configs/t022_ppo.yaml; same as t022-ppo-s0/s1)
#   vla-b03 -> run_t024.sh vla 0.3 <seed>        (T024 PPO + VLA Phi beta 0.3; same as t024-vla-b03-s0)
# Only actor.seed / env.train.seed change (run.sh sets both from <seed>); env.eval.seed=1000 and the 128 held-out
# eval ids are fixed by the config -> paired with the existing runs.
# usage: vla_residual/run_t030.sh <grpo|ppo|vla-b03> <seed> [extra hydra overrides (smoke only)]
set -uo pipefail
COND=${1:?cond}; SEED=${2:?seed}; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
# human instruction (2026-11-20): T030 runs on physical GPU 0 ONLY (GPU 1 and 3 forbidden).
if [ "${CUDA_VISIBLE_DEVICES:-none}" != 0 ]; then
  echo "[t030] REFUSE: CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}' (allowed: 0)"; exit 3
fi
export RUN_ID=${RUN_ID:-t030-$COND-s$SEED}
export RUN_DIR=${RUN_DIR:-/workspace/stochrl_t030/scratch/$RUN_ID}
case $COND in
  grpo|ppo) exec "$HERE/run.sh" "$COND" "$SEED" "$@" ;;
  vla-b03)  T024_GPUS=0 exec "$HERE/run_t024.sh" vla 0.3 "$SEED" "$@" ;;
  *) echo "[t030] unknown cond $COND" >&2; exit 2 ;;
esac
