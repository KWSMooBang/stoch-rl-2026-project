#!/usr/bin/env bash
# T022: RLinf <algo> residual RL on frozen pi0.5 (PutOnPlateInScene25Main-v3), ONE GPU (physical 0 or 1 only).
# usage: vla_residual/run.sh <ppo|grpo|sac|crossq|dagger> <seed> [extra hydra overrides...]
# env:   RUN_DIR / RUN_ID (harness), default scratch /workspace/stochrl_t022/scratch/<run_id>
set -uo pipefail
ALGO=${1:?algo}; SEED=${2:?seed}; shift 2
HERE=$(cd "$(dirname "$0")" && pwd); WT=$(dirname "$HERE")
RLINF=$WT/third_party/RLinf
PY=${T022_PY:-/workspace/StochRL/.claude/worktrees/repcritic/.venv/bin/python}

# ---- GPU guard (human instruction: GPU 0/1 only, never GPU 3) ----
case "${CUDA_VISIBLE_DEVICES:-}" in
  0|1) ;;
  *) echo "[t022] REFUSE: CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}' (only physical GPU 0 or 1 allowed)" >&2; exit 3 ;;
esac
GPU=$CUDA_VISIBLE_DEVICES
case $ALGO in ppo|grpo|sac|crossq|dagger) ;; *) echo "unknown algo $ALGO" >&2; exit 2 ;; esac

RUN_ID=${RUN_ID:-t022-$ALGO-s$SEED}
RUN_DIR=${RUN_DIR:-/workspace/stochrl_t022/scratch/$RUN_ID}
mkdir -p "$RUN_DIR"
[ -e "$RLINF/rlinf/envs/sim/maniskill/assets" ] || ln -s /workspace/stochrl_t016/maniskill_assets "$RLINF/rlinf/envs/sim/maniskill/assets"
export EMBODIED_PATH=$RLINF/examples/embodiment
export PYTHONPATH=$RLINF:$WT${PYTHONPATH:+:$PYTHONPATH}
export RLINF_EXT_MODULE=vla_residual.rlinf_ext
export VR_LOG_DIR=$RUN_DIR
export VK_ICD_FILENAMES=$WT/repcritic/vulkan/nvidia_icd.json VK_DRIVER_FILES=$WT/repcritic/vulkan/nvidia_icd.json
export TOKENIZERS_PARALLELISM=false HF_HOME=${HF_HOME:-/cache/huggingface}
# NOTE: no expandable_segments (breaks CUDA IPC between Ray workers: pidfd_getfd not permitted)
export RC_POLICY_MB=${RC_POLICY_MB:-8}   # pi0.5 micro-batch (activation memory; whole-GPU peak ~19-21 GB at 16)
export RAY_DEDUP_LOGS=0 HYDRA_FULL_ERROR=1
export RAY_TMPDIR=/tmp/ray_t022_${RUN_ID}
CFGDIR=$HERE/configs; CFG=t022_$ALGO

COMMON=(
  '~cluster.component_placement'
  "+cluster.component_placement={actor:$GPU,env:$GPU,rollout:$GPU}"
  runner.logger.log_path=$RUN_DIR/rlinf_logs
  runner.logger.experiment_name=$RUN_ID
  actor.seed=$SEED
  env.train.seed=$SEED
  env.eval.seed=1000
)
printf '%s\n' "$ALGO" "$SEED" "$CFG" "${COMMON[@]}" "$@" > "$RUN_DIR/t022_overrides.txt"
echo "[t022] algo=$ALGO seed=$SEED gpu=$GPU cfg=$CFGDIR/$CFG run_dir=$RUN_DIR"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$GPU" | tee "$RUN_DIR/gpu_baseline.txt"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU"; sleep 5; done ) > "$RUN_DIR/vram_trace.txt" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

# RLinf b665e18 fix from T021 (DAgger MLP sft loss reduction); idempotent
for pf in "$WT"/rlinf_algos/patches/*.patch; do
  if git -C "$RLINF" apply --check "$pf" 2>/dev/null; then git -C "$RLINF" apply "$pf" && echo "[t022] applied $(basename "$pf")";
  elif git -C "$RLINF" apply --reverse --check "$pf" 2>/dev/null; then echo "[t022] already applied $(basename "$pf")";
  else echo "[t022] patch does not apply: $pf" >&2; exit 4; fi
done
git -C "$RLINF" diff > "$RUN_DIR/rlinf_patch.diff"
git -C "$WT" rev-parse HEAD > "$RUN_DIR/t022_git_sha" 2>/dev/null || true

# ---- dedicated Ray head for this run (T021: ray.init(address=auto) joins any cluster on the machine) ----
RAY_BIN=$(dirname "$PY")/ray
export RAY_AUTH_MODE=token
RAY_PORT=$("$PY" -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])')
"$RAY_BIN" start --head --block --port="$RAY_PORT" --temp-dir="$RAY_TMPDIR" \
  --include-dashboard=false --disable-usage-stats --num-gpus=1 > "$RUN_DIR/ray_head.log" 2>&1 &
RAYPID=$!
trap 'kill $SAMPLER 2>/dev/null; kill -TERM $RAYPID 2>/dev/null; wait $RAYPID 2>/dev/null' EXIT
RAY_ADDRESS=""
for i in $(seq 1 60); do
  RAY_ADDRESS=$(grep -o "ray start --address='[^']*'" "$RUN_DIR/ray_head.log" 2>/dev/null | head -1 | sed "s/.*='//; s/'//")
  [ -n "$RAY_ADDRESS" ] && break; sleep 2
done
[ -n "$RAY_ADDRESS" ] && [[ $RAY_ADDRESS == *:$RAY_PORT ]] || { echo "[t022] ray head failed (addr='$RAY_ADDRESS')" >&2; exit 5; }
export RAY_ADDRESS
echo "[t022] ray head $RAY_ADDRESS pid=$RAYPID"

date +%s > "$RUN_DIR/t_start"
cd "$RLINF"
"$PY" "$EMBODIED_PATH/train_embodied_agent.py" --config-path "$CFGDIR" --config-name "$CFG" "${COMMON[@]}" "$@"
RC=$?
date +%s > "$RUN_DIR/t_end"
kill $SAMPLER 2>/dev/null
"$PY" "$HERE/make_metrics.py" --run_dir "$RUN_DIR" --algo "$ALGO" --seed "$SEED" --cfg "$CFGDIR/$CFG.yaml" --rc $RC || true
exit $RC
