#!/usr/bin/env bash
# T021: run one RLinf algorithm on ManiSkill PickCube (MLP, state obs) on ONE GPU (physical 0 or 1 only).
# usage: run_algo.sh <algo: ppo|grpo|sac|crossq|dagger> <seed> [extra hydra overrides...]
# env:   RUN_DIR (harness sets it; default /workspace/stochrl_t021/scratch/<algo>-s<seed>), RUN_ID
set -uo pipefail
ALGO=${1:?algo}; SEED=${2:?seed}; shift 2
HERE=$(cd "$(dirname "$0")" && pwd); WT=$(dirname "$HERE")
RLINF=$WT/third_party/RLinf
PY=${T021_PY:-/workspace/StochRL/.claude/worktrees/repcritic/.venv/bin/python}

# ---- GPU guard (human instruction: GPU 0/1 only, never GPU 3) ----
case "${CUDA_VISIBLE_DEVICES:-}" in
  0|1) ;;
  *) echo "[t021] REFUSE: CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}' (only physical GPU 0 or 1 allowed)" >&2; exit 3 ;;
esac
GPU=$CUDA_VISIBLE_DEVICES

RUN_ID=${RUN_ID:-t021-$ALGO-s$SEED}
RUN_DIR=${RUN_DIR:-/workspace/stochrl_t021/scratch/$RUN_ID}
mkdir -p "$RUN_DIR"
export EMBODIED_PATH=$RLINF/examples/embodiment
export PYTHONPATH=$RLINF${PYTHONPATH:+:$PYTHONPATH}
export VK_ICD_FILENAMES=$WT/repcritic/vulkan/nvidia_icd.json VK_DRIVER_FILES=$WT/repcritic/vulkan/nvidia_icd.json
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export RAY_DEDUP_LOGS=0 HYDRA_FULL_ERROR=1
export RAY_TMPDIR=/tmp/ray_t021_${RUN_ID}

case $ALGO in
  ppo)    CFGDIR=$EMBODIED_PATH/config; CFG=maniskill_ppo_mlp ;;
  sac)    CFGDIR=$EMBODIED_PATH/config; CFG=maniskill_sac_mlp ;;
  crossq) CFGDIR=$EMBODIED_PATH/config; CFG=maniskill_crossq_mlp ;;
  dagger) CFGDIR=$EMBODIED_PATH/config; CFG=maniskill_dagger_mlp ;;
  grpo)   CFGDIR=$HERE/configs;         CFG=maniskill_grpo_mlp ;;
  *) echo "unknown algo $ALGO" >&2; exit 2 ;;
esac

# Common overrides (fairness): same eval env count / eval seed, train seed = SEED, no videos.
# NOTE: RLinf maps cluster.component_placement indices to *physical* GPU ids (it enumerates all
# node GPUs via NVML and sets the workers' CUDA_VISIBLE_DEVICES itself; measured: outer CVD=1 +
# placement 0 -> load on physical GPU 0). So the placement must be the physical id = $GPU.
COMMON=(
  '~cluster.component_placement'
  "+cluster.component_placement={actor:$GPU,env:$GPU,rollout:$GPU}"
  runner.logger.log_path=$RUN_DIR/rlinf_logs
  runner.logger.experiment_name=$RUN_ID
  actor.seed=$SEED
  env.train.seed=$SEED
  env.eval.seed=${T021_EVAL_SEED:-1000}
  env.eval.total_num_envs=${T021_EVAL_ENVS:-128}
  env.eval.video_cfg.save_video=False
)
printf '%s\n' "$ALGO" "$SEED" "$CFG" "${COMMON[@]}" "$@" > "$RUN_DIR/t021_overrides.txt"
echo "[t021] algo=$ALGO seed=$SEED gpu=$GPU cfg=$CFGDIR/$CFG run_dir=$RUN_DIR"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$GPU" | tee "$RUN_DIR/gpu_baseline.txt"

# peak-VRAM sampler (whole-GPU memory.used, baseline recorded above)
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU"; sleep 5; done ) > "$RUN_DIR/vram_trace.txt" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

# RLinf b665e18 fixes needed by T021 (idempotent; recorded in the run dir)
for pf in "$HERE"/patches/*.patch; do
  if git -C "$RLINF" apply --check "$pf" 2>/dev/null; then git -C "$RLINF" apply "$pf" && echo "[t021] applied $(basename "$pf")";
  elif git -C "$RLINF" apply --reverse --check "$pf" 2>/dev/null; then echo "[t021] already applied $(basename "$pf")";
  else echo "[t021] patch does not apply: $pf" >&2; exit 4; fi
done
git -C "$RLINF" diff > "$RUN_DIR/rlinf_patch.diff"

# ---- dedicated Ray head for this run ----
# RLinf calls ray.init(address="auto"); Ray then joins ANY gcs_server found on the machine
# (measured: a 2nd run attached to another run's cluster and died with it). So start our own
# head and pin it via RAY_ADDRESS (checked first by Ray's address discovery).
RAY_BIN=$(dirname "$PY")/ray
export RAY_AUTH_MODE=token   # head (ray start) enables token auth; driver/workers must use the same token
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
[ -n "$RAY_ADDRESS" ] && [[ $RAY_ADDRESS == *:$RAY_PORT ]] || { echo "[t021] ray head failed (addr='$RAY_ADDRESS')" >&2; exit 5; }
export RAY_ADDRESS
echo "[t021] ray head $RAY_ADDRESS pid=$RAYPID"

date +%s > "$RUN_DIR/t_start"
cd "$RLINF"
"$PY" "$EMBODIED_PATH/train_embodied_agent.py" --config-path "$CFGDIR" --config-name "$CFG" "${COMMON[@]}" "$@"
RC=$?
date +%s > "$RUN_DIR/t_end"
kill $SAMPLER 2>/dev/null
"$PY" "$HERE/make_metrics.py" --run_dir "$RUN_DIR" --algo "$ALGO" --seed "$SEED" --cfg "$CFGDIR/$CFG.yaml" --rc $RC || true
exit $RC
