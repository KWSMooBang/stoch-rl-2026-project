#!/usr/bin/env bash
# T032 submit chain, GPU 0 ONLY (human instruction 2026-11-20: cap 4.0 GPU-h incl. validation, GPU 1/3 forbidden).
# One run at a time: submit only when GPU 0 is free (< 1000 MiB) and no t032-* run is queued/running (the harness
# picks the lowest free index -> GPU 0). run_t032.sh refuses any other GPU (exit 3) -> resubmitted as <run_id>-rN.
# Budget guard: skip a run if used (results.tsv t032-* gpu_h + T032_PRE_GPUH) + estimate > T032_BUDGET.
# usage: T032_PRE_GPUH=<validation+smoke GPU-h> nohup setsid vla_residual/submit_t032.sh \
#          "t032-grpo-e0-s0:e0:0:1.25" "t032-grpo-e0-s1:e0:1:1.25" "t032-grpo-e0-s2:e0:2:1.25" > chain.log 2>&1 &
set -uo pipefail
WT=$(cd "$(dirname "$0")/.." && pwd)
cd "$WT"
OUT=/outputs/StochRL
BUDGET=${T032_BUDGET:-4.0}
PRE=${T032_PRE_GPUH:?validation + smoke GPU-h spent outside the harness}

log() { echo "[$(date -Is)] $*"; }
free_0() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '$1==0 && $2<1000 {f=1} END {print f+0}'; }
count() {  # count <status regex> over t032-* runs
  local n=0 s
  for s in "$OUT"/runs/t032-*/STATUS; do
    [ -f "$s" ] || continue
    grep -qE "^($1)$" "$s" && n=$((n+1))
  done
  echo $n
}
used_gpuh() { awk -F'\t' -v pre="$PRE" '$1 ~ /^t032-/ {s+=$5} END {printf "%.2f", s+pre}' "$OUT/results.tsv" 2>/dev/null; }

log "start T032 chain (budget $BUDGET, pre $PRE)"
for spec in "$@"; do
  IFS=: read -r BASE COND SEED HOURS <<< "$spec"
  n=0
  while true; do
    until [ "$(count 'queued|running')" -eq 0 ] && [ "$(free_0)" -eq 1 ]; do sleep 60; done
    sleep 20  # let the harness append the previous run's results.tsv row
    U=$(used_gpuh)
    if awk -v u="$U" -v h="$HOURS" -v b="$BUDGET" 'BEGIN{exit !(u+h>b)}'; then
      log "SKIP $BASE: budget used $U + est $HOURS > $BUDGET"; break
    fi
    ID=$BASE; [ $n -gt 0 ] && ID=$BASE-r$n
    log "submit $ID (used $U GPU-h, est $HOURS)"
    RUN_HOURS=$(awk -v h="$HOURS" 'BEGIN{printf "%.2f", h*1.4+0.2}') /harness/launch.sh "$ID" 1 vla_residual/run_t032.sh "$COND" "$SEED"
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
