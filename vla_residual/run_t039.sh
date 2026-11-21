#!/usr/bin/env bash
# T039: noise-space steering. Same GRPO config / obs (E2: VLM-17 feature | qpos | base chunk | t) / seeds / eval as
# t022-grpo-s0 and t030-grpo-s{1,2}; the 7-D policy output now shifts pi0.5's initial flow noise
#   z = z_ref + beta * tanh(u)   on model action dims 0..6, broadcast over the H=8 horizon   (u = 0 -> B0 exactly)
# instead of adding alpha * tanh(u) to the action (vla_residual.noise_steer.mode=add, beta=2.0; design: plan_t039_ko.md).
# usage: vla_residual/run_t039.sh <seed> [extra hydra overrides (smoke only)]
set -uo pipefail
SEED=${1:?seed}; shift 1
HERE=$(cd "$(dirname "$0")" && pwd)
BETA=${T039_BETA:-2.0}
# human instruction (2026-11-21): T039 runs on physical GPU 0 ONLY (GPU 1 and 3 forbidden).
if [ "${CUDA_VISIBLE_DEVICES:-none}" != 0 ]; then
  echo "[t039] REFUSE: CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}' (allowed: 0)"; exit 3
fi
export RUN_ID=${RUN_ID:-t039-noise-s$SEED}
export RUN_DIR=${RUN_DIR:-/workspace/stochrl_t039/scratch/$RUN_ID}
exec "$HERE/run.sh" grpo "$SEED" \
  env.train.vla_residual.noise_steer.mode=add env.eval.vla_residual.noise_steer.mode=add \
  env.train.vla_residual.noise_steer.beta=$BETA env.eval.vla_residual.noise_steer.beta=$BETA "$@"
