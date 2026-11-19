#!/usr/bin/env bash
# T022 submit chain: harness runs one at a time per free GPU 0/1 (max 2 concurrent), never GPU 3.
# The harness takes the lowest-index GPU with < 1000 MiB used; GPU 3 is often free, so we only submit when
# GPU 0 or 1 is free and nothing of ours is queued. run.sh refuses (exit 3) any other GPU; a refused run is
# resubmitted as <run_id>-rN. Budget guard: skip a run if used + estimate > T022_BUDGET (GPU-h).
# usage: nohup setsid vla_residual/submit_chain.sh "ppo:0:1.5" "grpo:0:1.5" ... > chain.log 2>&1 &
set -uo pipefail
WT=$(cd "$(dirname "$0")/.." && pwd)
cd "$WT"
OUT=/outputs/StochRL
BUDGET=${T022_BUDGET:-10}
PRE=${T022_PRE_GPUH:-0}       # GPU-h spent outside the harness (validation, smoke)
MAXPAR=${T022_MAXPAR:-2}

log() { echo "[$(date -Is)] $*"; }
free_01() {  # prints "1" if GPU 0 or 1 has < 1000 MiB used
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '($1==0||$1==1) && $2<1000 {f=1} END {print f+0}'
}
ours_active() {
  local n=0 s
  for s in "$OUT"/runs/t022-*/STATUS; do
    [ -f "$s" ] || continue
    case $(cat "$s") in queued|running) n=$((n+1)) ;; esac
  done
  echo $n
}
ours_queued() {
  local n=0 s
  for s in "$OUT"/runs/t022-*/STATUS; do
    [ -f "$s" ] || continue
    [ "$(cat "$s")" = queued ] && n=$((n+1))
  done
  echo $n
}
used_gpuh() {
  awk -F'\t' -v pre="$PRE" '$1 ~ /^t022-/ {s+=$5} END {printf "%.2f", s+pre}' "$OUT/results.tsv" 2>/dev/null
}

for spec in "$@"; do
  IFS=: read -r ALGO SEED HOURS <<< "$spec"
  BASE=t022-$ALGO-s$SEED
  n=0
  while true; do
    # wait for a slot: < MAXPAR active, none of ours queued, GPU 0 or 1 free
    until [ "$(ours_active)" -lt "$MAXPAR" ] && [ "$(ours_queued)" -eq 0 ] && [ "$(free_01)" -eq 1 ]; do sleep 60; done
    U=$(used_gpuh)
    if awk -v u="$U" -v h="$HOURS" -v b="$BUDGET" 'BEGIN{exit !(u+h>b)}'; then
      log "SKIP $BASE: budget used $U + est $HOURS > $BUDGET"; break
    fi
    ID=$BASE; [ $n -gt 0 ] && ID=$BASE-r$n
    log "submit $ID (used $U GPU-h, est $HOURS)"
    RUN_HOURS=$(awk -v h="$HOURS" 'BEGIN{printf "%.2f", h*1.4+0.2}') /harness/launch.sh "$ID" 1 vla_residual/run.sh "$ALGO" "$SEED"
    # wait until it leaves the queue, then until it is clearly running on GPU 0/1 or refused
    while [ "$(cat "$OUT/runs/$ID/STATUS" 2>/dev/null)" = queued ]; do sleep 30; done
    sleep 90
    G=$(cat "$OUT/runs/$ID/gpus" 2>/dev/null)
    if grep -q "REFUSE" "$OUT/runs/$ID/stdout.log" 2>/dev/null; then
      log "$ID refused on GPU $G -> resubmit"; n=$((n+1)); [ $n -gt 20 ] && { log "give up $BASE"; break; }
      sleep 120; continue
    fi
    log "$ID running on GPU $G"
    break
  done
done
log "chain done"
