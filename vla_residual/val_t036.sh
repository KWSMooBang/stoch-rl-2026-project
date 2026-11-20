#!/usr/bin/env bash
# T036 pre-training checks outside the harness (single process, physical GPU 0 only).
# usage: vla_residual/val_t036.sh <cpu|unit|b0> [extra eval_t036.py args]
set -uo pipefail
MODE=${1:?mode}; shift
HERE=$(cd "$(dirname "$0")" && pwd); WT=$(dirname "$HERE")
PY=${T022_PY:-/workspace/StochRL/.claude/worktrees/repcritic/.venv/bin/python}
RLINF=$WT/third_party/RLinf
[ -e "$RLINF/rlinf/envs/sim/maniskill/assets" ] || ln -s /workspace/stochrl_t016/maniskill_assets "$RLINF/rlinf/envs/sim/maniskill/assets"
export PYTHONPATH=$RLINF:$WT${PYTHONPATH:+:$PYTHONPATH}
export VK_ICD_FILENAMES=$WT/repcritic/vulkan/nvidia_icd.json VK_DRIVER_FILES=$WT/repcritic/vulkan/nvidia_icd.json
export TOKENIZERS_PARALLELISM=false HF_HOME=${HF_HOME:-/cache/huggingface} RC_POLICY_MB=${RC_POLICY_MB:-8}
[ "$MODE" = cpu ] && export CUDA_VISIBLE_DEVICES= || export CUDA_VISIBLE_DEVICES=0
cd "$WT"
T0=$(date +%s)
"$PY" -m vla_residual.eval_t036 --mode "$MODE" --out /outputs/StochRL/t036/val "$@"
RC=$?
echo "[val_t036] $MODE rc=$RC wall=$(( $(date +%s) - T0 ))s"
exit $RC
