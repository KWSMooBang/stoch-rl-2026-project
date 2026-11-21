#!/usr/bin/env bash
# T037 submit chain, GPU 0 ONLY (human instruction 2026-11-21: cap 4.2 GPU-h incl. validation, GPU 1/3 forbidden,
# "first run decides": only s0 is submitted; s1 / s2 follow only after L1 creates /outputs/StochRL/t037/GO).
#   1. wait until the T036 chain has logged "chain done", no t036-* .. t039-* run is queued/running and GPU 0 is free
#      (< 1000 MiB; the harness picks the lowest free index -> GPU 0). run_t037.sh refuses any other GPU (exit 3)
#      -> resubmitted as -rN.
#   2. submit t037-e0v-s0, wait until it has ended, then wait for the GO or STOP file (written by L1 from the
#      pre-registered keep rule: end >= 0.58 and once >= 0.68 on s0).
#   3. GO -> t037-e0v-s1, t037-e0v-s2 one at a time (same waiting rule). STOP -> end.
# Budget guard: skip a run if used (results.tsv t037-* gpu_h + T037_PRE_GPUH) + estimate > T037_BUDGET (drops s2 first).
# usage: T037_PRE_GPUH=<validation+smoke GPU-h> nohup setsid vla_residual/submit_t037.sh > /outputs/StochRL/t037/chain.log 2>&1 &
set -uo pipefail
WT=$(cd "$(dirname "$0")/.." && pwd)
cd "$WT"
OUT=/outputs/StochRL
BUDGET=${T037_BUDGET:-4.2}
PRE=${T037_PRE_GPUH:?validation + smoke GPU-h spent outside the harness}
EST=${T037_EST:-1.25}
PREV_CHAIN=${T037_PREV_CHAIN:-$OUT/t036/chain.log}

log() { echo "[$(date -Is)] $*"; }
free_0() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '$1==0 && $2<1000 {f=1} END {print f+0}'; }
busy() {  # number of t036-* .. t039-* runs that are queued or running
  local n=0 s
  for s in "$OUT"/runs/t036-*/STATUS "$OUT"/runs/t037-*/STATUS "$OUT"/runs/t038-*/STATUS "$OUT"/runs/t039-*/STATUS; do
    [ -f "$s" ] || continue
    grep -qE "^(queued|running)$" "$s" && n=$((n+1))
  done
  echo $n
}
used_gpuh() { awk -F'\t' -v pre="$PRE" '$1 ~ /^t037-/ {s+=$5} END {printf "%.2f", s+pre}' "$OUT/results.tsv" 2>/dev/null; }

LAST_ID=
submit() {  # submit <base run id> <seed>; sets LAST_ID (empty if skipped)
  local BASE=$1 SEED=$2 n=0 U ID G
  LAST_ID=
  while true; do
    until [ "$(busy)" -eq 0 ] && [ "$(free_0)" -eq 1 ]; do sleep 60; done
    sleep 20  # let the harness append the previous run's results.tsv row
    [ "$(busy)" -eq 0 ] && [ "$(free_0)" -eq 1 ] || continue
    U=$(used_gpuh)
    if awk -v u="$U" -v h="$EST" -v b="$BUDGET" 'BEGIN{exit !(u+h>b)}'; then
      log "SKIP $BASE: budget used $U + est $EST > $BUDGET"; return
    fi
    ID=$BASE; [ $n -gt 0 ] && ID=$BASE-r$n
    log "submit $ID (used $U GPU-h, est $EST)"
    RUN_HOURS=$(awk -v h="$EST" 'BEGIN{printf "%.2f", h*1.4+0.2}') /harness/launch.sh "$ID" 1 vla_residual/run_t037.sh e0v "$SEED"
    while [ "$(cat "$OUT/runs/$ID/STATUS" 2>/dev/null)" = queued ]; do sleep 30; done
    sleep 90
    G=$(cat "$OUT/runs/$ID/gpus" 2>/dev/null)
    if grep -q "REFUSE" "$OUT/runs/$ID/stdout.log" 2>/dev/null; then
      log "$ID refused on GPU $G -> resubmit"; n=$((n+1)); [ $n -gt 20 ] && { log "give up $BASE"; return; }
      sleep 120; continue
    fi
    log "$ID running on GPU $G"
    LAST_ID=$ID
    return
  done
}

log "start T037 chain (budget $BUDGET, pre $PRE, est $EST per run)"
until grep -q "chain done" "$PREV_CHAIN" 2>/dev/null; do sleep 60; done
log "previous chain done ($PREV_CHAIN)"
submit t037-e0v-s0 0
if [ -z "$LAST_ID" ]; then log "s0 not submitted"; log "chain done"; exit 0; fi
while grep -qE "^(queued|running)$" "$OUT/runs/$LAST_ID/STATUS" 2>/dev/null; do sleep 60; done
log "$LAST_ID ended: STATUS=$(cat "$OUT/runs/$LAST_ID/STATUS" 2>/dev/null) -> waiting for $OUT/t037/GO or STOP (L1)"
until [ -e "$OUT/t037/GO" ] || [ -e "$OUT/t037/STOP" ]; do sleep 60; done
if [ -e "$OUT/t037/STOP" ]; then
  log "STOP file found -> no s1 / s2"
else
  log "GO file found -> s1, s2"
  submit t037-e0v-s1 1
  submit t037-e0v-s2 2
fi
log "chain done"
