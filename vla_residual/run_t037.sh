#!/usr/bin/env bash
# T037: E0v = minimal-information residual. GRPO residual config of E2 / E0 (t022_grpo.yaml) with the observation
#     states = [ 0 x 2048 | qpos 8 | base chunk 35 | t/16 | V_hat_end ]            (dim 2093)
#   zero_vlm=True   VLM feature block = 0 (T032 E0)
#   vhat_obs=True   one extra column: frozen V_hat_end (T036 predictor; VLM17 pooled feature -> success_at_end, Platt)
#                   of the decision state, from the same pi0.5 forward (no extra forward)
#   gate.mode=one   g = 1: the executed action is the E2 / E0 formula clip(a_base + alpha tanh(u)). The gate block only
#                   loads the predictor and logs V_hat per decision (gate_decisions.jsonl).
#   actor.model.obs_dim=2093 (plain RLinf MLP; no gate head, action_dim 7)
# Everything else (GRPO, alpha, sparse reward, lr, epochs = env steps, eval seed 1000, held-out 128) is unchanged.
# usage: vla_residual/run_t037.sh e0v <seed> [extra hydra overrides (smoke only)]
set -uo pipefail
COND=${1:?cond}; SEED=${2:?seed}; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
# human instruction (2026-11-21): T037 runs on physical GPU 0 ONLY (GPU 1 and 3 forbidden).
if [ "${CUDA_VISIBLE_DEVICES:-none}" != 0 ]; then
  echo "[t037] REFUSE: CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}' (allowed: 0)"; exit 3
fi
export RUN_ID=${RUN_ID:-t037-$COND-s$SEED}
export RUN_DIR=${RUN_DIR:-/workspace/stochrl_t037/scratch/$RUN_ID}
PY=${T022_PY:-/workspace/StochRL/.claude/worktrees/repcritic/.venv/bin/python}
both() { echo "env.train.vla_residual.$1" "env.eval.vla_residual.$1"; }
case $COND in
  e0v) "$HERE/run.sh" grpo "$SEED" $(both zero_vlm=True) $(both vhat_obs=True) $(both gate.mode=one) $(both gate.vhat=end) \
         actor.model.obs_dim=2093 "$@" ;;
  *) echo "[t037] unknown cond $COND" >&2; exit 2 ;;
esac
RC=$?
"$PY" "$HERE/make_metrics_t036.py" --run_dir "$RUN_DIR" --cond "$COND" --task T037 --tau 0.41 --temp 0.1047 || true
exit $RC
