#!/usr/bin/env bash
# T035: base-anchored GRPO = E2 GRPO residual config (t022_grpo.yaml; same as t022-grpo-s0 / t030-grpo-s1,s2) with
#   - every train group of 8 = 2 frozen-base members (executed residual 0) + 6 residual members
#   - pi0.5 noise shared inside a group per (episode id, decision)                       (CRN)
#   - A_i = (R_i - mean base score) / (std of the 6 residual scores + eps), base members out of the policy loss
# Everything else (obs incl. VLA feature, alpha, lr, epochs = total env steps, eval seed 1000, held-out 128) unchanged.
#   anchor -> run.sh grpo <seed> + the four overrides below
#   e2     -> run.sh grpo <seed>                 (reference path, not used for T035 runs)
# usage: vla_residual/run_t035.sh <anchor|e2> <seed> [extra hydra overrides (smoke only)]
set -uo pipefail
COND=${1:?cond}; SEED=${2:?seed}; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
# human instruction (2026-11-21): T035 runs on physical GPU 0 ONLY (GPU 1 and 3 forbidden).
if [ "${CUDA_VISIBLE_DEVICES:-none}" != 0 ]; then
  echo "[t035] REFUSE: CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}' (allowed: 0)"; exit 3
fi
export RUN_ID=${RUN_ID:-t035-$COND-s$SEED}
export RUN_DIR=${RUN_DIR:-/workspace/stochrl_t035/scratch/$RUN_ID}
NB=${T035_N_BASE:-2}
case $COND in
  anchor) exec "$HERE/run.sh" grpo "$SEED" env.train.vla_residual.anchor_n_base="$NB" env.train.vla_residual.anchor_crn=True \
            algorithm.adv_type=grpo_anchor "+algorithm.advantage_mode='anchor_nbase=$NB'" "$@" ;;
  e2)     exec "$HERE/run.sh" grpo "$SEED" "$@" ;;
  *) echo "[t035] unknown cond $COND" >&2; exit 2 ;;
esac
