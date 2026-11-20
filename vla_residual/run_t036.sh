#!/usr/bin/env bash
# T036: competence-gated residual = E2 GRPO residual config (t022_grpo.yaml; same as t022-grpo-s0 / t030-grpo-s1,s2)
# with the executed residual multiplied by a per-decision gate g in [0,1] from the frozen V_hat_end
# (VLM17 feature -> success_at_end, all states incl. post-success; vla_residual/fit_vhat_end.py; L1 decision 2026-11-21):
#     a = clip(a_base + alpha * g * tanh(u))
#   g1   G1 fixed   g = sigmoid((tau - V_hat) / temp)                    (env only; policy / obs / loss = E2)
#   g2   G2 learned g = sigmoid(z), z = 8th action dim; mean z = S (p_v V_hat + p_t t/16 + p_b), init = G1
#   one  g = 1 through the gate code path (smoke / bitwise reference)    e2  plain E2 (reference path)
#   diag evaluation only: stored E2 checkpoints with the G1 gate (V_hat_end and T023 V_hat) and without it (vla_residual/eval_t036.py --mode diag)
# Everything else (reward, alpha, lr, epochs = env steps, eval seed 1000, held-out 128) is unchanged.
# usage: vla_residual/run_t036.sh <g1|g2|one|e2|diag> <seed> [extra hydra overrides (smoke only)]
set -uo pipefail
COND=${1:?cond}; SEED=${2:?seed}; shift 2
HERE=$(cd "$(dirname "$0")" && pwd); WT=$(dirname "$HERE")
# human instruction (2026-11-21): T036 runs on physical GPU 0 ONLY (GPU 1 and 3 forbidden).
if [ "${CUDA_VISIBLE_DEVICES:-none}" != 0 ]; then
  echo "[t036] REFUSE: CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}' (allowed: 0)"; exit 3
fi
export RUN_ID=${RUN_ID:-t036-$COND-s$SEED}
export RUN_DIR=${RUN_DIR:-/workspace/stochrl_t036/scratch/$RUN_ID}
PY=${T022_PY:-/workspace/StochRL/.claude/worktrees/repcritic/.venv/bin/python}
TAU=0.41; TEMP=0.1047; VHAT=end  # vla_residual/gate.py end (train split only) -> /outputs/StochRL/t036/gate_select_end.json
G2_SCALE=100; G2_LOGSTD=0.0    # gate-logit step multiplier S and initial log-std of the gate logit z
gate() { echo "env.train.vla_residual.gate.mode=$1" "env.eval.vla_residual.gate.mode=$1" \
              "env.train.vla_residual.gate.vhat=$VHAT" "env.eval.vla_residual.gate.vhat=$VHAT" \
              "env.train.vla_residual.gate.tau=$TAU" "env.eval.vla_residual.gate.tau=$TAU" \
              "env.train.vla_residual.gate.temp=$TEMP" "env.eval.vla_residual.gate.temp=$TEMP"; }
case $COND in
  g1)  "$HERE/run.sh" grpo "$SEED" $(gate fixed) "$@" ;;
  one) "$HERE/run.sh" grpo "$SEED" $(gate one) "$@" ;;
  g2)  "$HERE/run.sh" grpo "$SEED" $(gate learned) actor.model.obs_dim=2093 actor.model.action_dim=8 \
         "+actor.model.vr_gate={tau:$TAU,temp:$TEMP,scale:$G2_SCALE,init_logstd:$G2_LOGSTD}" "$@" ;;
  e2)  "$HERE/run.sh" grpo "$SEED" "$@" ;;
  diag)
    mkdir -p "$RUN_DIR"
    RLINF=$WT/third_party/RLinf
    [ -e "$RLINF/rlinf/envs/sim/maniskill/assets" ] || ln -s /workspace/stochrl_t016/maniskill_assets "$RLINF/rlinf/envs/sim/maniskill/assets"
    export PYTHONPATH=$RLINF:$WT${PYTHONPATH:+:$PYTHONPATH}
    export VK_ICD_FILENAMES=$WT/repcritic/vulkan/nvidia_icd.json VK_DRIVER_FILES=$WT/repcritic/vulkan/nvidia_icd.json
    export TOKENIZERS_PARALLELISM=false HF_HOME=${HF_HOME:-/cache/huggingface} RC_POLICY_MB=${RC_POLICY_MB:-8}
    cd "$WT" && "$PY" -m vla_residual.eval_t036 --mode diag --out "$RUN_DIR" --json metrics.json "$@"
    exit $? ;;
  *) echo "[t036] unknown cond $COND" >&2; exit 2 ;;
esac
RC=$?
"$PY" "$HERE/make_metrics_t036.py" --run_dir "$RUN_DIR" --cond "$COND" --tau $TAU --temp $TEMP --scale $G2_SCALE || true
exit $RC
