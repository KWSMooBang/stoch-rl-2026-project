#!/usr/bin/env bash
# T038: committed residual = E2 (GRPO + VLA feature, same config as t022-grpo-s0 / t030-grpo-s1,s2) with the residual
# sampled every k decisions and re-executed in between (policy output and exploration noise alike).
#   k>1 : env.{train,eval}.vla_residual.commit_k=k  (adds the c_rem/k column -> actor.model.obs_dim=2093)
#         algorithm.adv_type=grpo_commit, algorithm.advantage_mode='commit_k=k' (loss only at sample decisions)
#   k=1 : same flags with k=1 -> E2 (no extra column, obs_dim 2092, GRPO inputs untouched)
# Everything else (alpha, sparse reward, lr 1e-4, 27 epochs, eval held-out 128 seed 1000) is unchanged.
# usage: vla_residual/run_t038.sh <k> <seed> [extra hydra overrides (smoke only)]
set -uo pipefail
K=${1:?k}; SEED=${2:?seed}; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
# human instruction (2026-11-21): T038 runs on physical GPU 0 ONLY (GPU 1 and 3 forbidden).
if [ "${CUDA_VISIBLE_DEVICES:-none}" != 0 ]; then
  echo "[t038] REFUSE: CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}' (allowed: 0)"; exit 3
fi
case $K in ''|*[!0-9]*) echo "[t038] bad k '$K'" >&2; exit 2 ;; esac
export RUN_ID=${RUN_ID:-t038-k$K-s$SEED}
export RUN_DIR=${RUN_DIR:-/workspace/stochrl_t038/scratch/$RUN_ID}
OBS=2092; [ "$K" -gt 1 ] && OBS=2093
exec "$HERE/run.sh" grpo "$SEED" env.train.vla_residual.commit_k="$K" env.eval.vla_residual.commit_k="$K" \
  actor.model.obs_dim=$OBS algorithm.adv_type=grpo_commit "+algorithm.advantage_mode='commit_k=$K'" "$@"
