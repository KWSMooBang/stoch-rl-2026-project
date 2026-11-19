#!/usr/bin/env bash
# T028: T022 GRPO (configs/t022_grpo.yaml, identical structure / alpha / obs / hyper-params / env steps / eval ids)
# with the T024 PBRS on the TRAIN env only: r' = r + beta (gamma Phi(s') - Phi(s)), Phi(s') = 1 on success termination,
# truncation keeps V_hat(s'). gamma = 0.96 = algorithm.gamma (per decision). Potential: T023 frozen Platt predictor.
# NOTE: RLinf GRPO scores an episode by the UNdiscounted sum of its rewards up to the first done (design_ko.md §1).
# usage: vla_residual/run_t028.sh <vla|raw_q|time> <beta> <seed> [extra hydra overrides]
set -uo pipefail
POT=${1:?potential}; BETA=${2:?beta}; SEED=${3:?seed}; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
case $POT in vla|raw_q|time) ;; *) echo "unknown potential $POT" >&2; exit 2 ;; esac
# human instruction (T028): physical GPU 0 ONLY (GPU 1 and 3 forbidden). Override list via T028_GPUS.
case " ${T028_GPUS:-0} " in
  *" ${CUDA_VISIBLE_DEVICES:-none} "*) ;;
  *) echo "[t028] REFUSE: CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}' (allowed: ${T028_GPUS:-0})"; exit 3 ;;
esac
export RUN_ID=${RUN_ID:-t028-grpo-$POT-b$BETA-s$SEED}
export RUN_DIR=${RUN_DIR:-/workspace/stochrl_t028/scratch/$RUN_ID}
exec "$HERE/run.sh" grpo "$SEED" \
  +env.train.vla_residual.pbrs.potential="$POT" \
  +env.train.vla_residual.pbrs.beta="$BETA" \
  +env.train.vla_residual.pbrs.gamma=0.96 \
  +env.train.vla_residual.pbrs.phi_term=1.0 \
  "$@"
