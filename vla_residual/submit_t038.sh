#!/usr/bin/env bash
# T038 submit chain, GPU 0 ONLY (human instruction 2026-11-21; cap 4.5 GPU-h incl. validation + smoke, GPU 1/3 forbidden).
# 1) s0 alone. 2) wait until L1 writes /outputs/StochRL/t038/GO or /outputs/StochRL/t038/STOP (pre-registered
#    criterion on s0's metrics.json). GO -> s1, s2 (same k); STOP -> end.
# One run at a time: submit only when GPU 0 is free (< 1000 MiB) and no t036-/t037-/t038-/t039- run is queued/running
# (the harness picks the lowest free index -> GPU 0). run_t038.sh refuses any other GPU (exit 3) -> resubmitted as -rN.
# Budget guard: skip a run if used (results.tsv t038-* gpu_h + T038_PRE_GPUH) + estimate > T038_BUDGET (s2 first).
# usage: T038_PRE_GPUH=<validation+smoke GPU-h> nohup setsid vla_residual/submit_t038.sh 3 > chain.log 2>&1 &
set -uo pipefail
K=${1:-3}
WT=$(cd "$(dirname "$0")/.." && pwd)
cd "$WT"
OUT=/outputs/StochRL
GATE=$OUT/t038
BUDGET=${T038_BUDGET:-4.5}
HOURS=${T038_EST_H:-1.3}
PRE=${T038_PRE_GPUH:?validation + smoke GPU-h spent outside the harness}

log() { echo "[$(date -Is)] $*"; }
free_0() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '$1==0 && $2<1000 {f=1} END {print f+0}'; }
busy() {  # queued/running StochRL runs of T036-T039
  local n=0 s
  for s in "$OUT"/runs/t03[6-9]-*/STATUS; do
    [ -f "$s" ] || continue
    grep -qE "^(queued|running)$" "$s" && n=$((n+1))
  done
  echo $n
}
used_gpuh() { awk -F'\t' -v pre="$PRE" '$1 ~ /^t038-/ {s+=$5} END {printf "%.2f", s+pre}' "$OUT/results.tsv" 2>/dev/null; }

submit() {  # submit <base_id> <seed>; returns after the run started on GPU 0 (or was skipped)
  local BASE=$1 SEED=$2 n=0 ID U G
  while true; do
    until [ "$(busy)" -eq 0 ] && [ "$(free_0)" -eq 1 ]; do sleep 60; done
    sleep 20  # let the harness append the previous run's results.tsv row
    U=$(used_gpuh)
    if awk -v u="$U" -v h="$HOURS" -v b="$BUDGET" 'BEGIN{exit !(u+h>b)}'; then
      log "SKIP $BASE: budget used $U + est $HOURS > $BUDGET"; return 1
    fi
    ID=$BASE; [ $n -gt 0 ] && ID=$BASE-r$n
    log "submit $ID (used $U GPU-h, est $HOURS)"
    RUN_HOURS=$(awk -v h="$HOURS" 'BEGIN{printf "%.2f", h*1.4+0.2}') /harness/launch.sh "$ID" 1 vla_residual/run_t038.sh "$K" "$SEED"
    while [ "$(cat "$OUT/runs/$ID/STATUS" 2>/dev/null)" = queued ]; do sleep 30; done
    sleep 90
    G=$(cat "$OUT/runs/$ID/gpus" 2>/dev/null)
    if grep -q "REFUSE" "$OUT/runs/$ID/stdout.log" 2>/dev/null; then
      log "$ID refused on GPU $G -> resubmit"; n=$((n+1)); [ $n -gt 20 ] && { log "give up $BASE"; return 1; }
      sleep 120; continue
    fi
    log "$ID running on GPU $G"
    LAST=$ID
    return 0
  done
}

log "start T038 chain k=$K (budget $BUDGET, pre $PRE, est $HOURS/run)"
LAST=
submit "t038-k$K-s0" 0 || { log "chain done (s0 not submitted)"; exit 0; }
log "waiting for $LAST to finish"
until grep -qE "^(done|failed|error|killed|timeout)" "$OUT/runs/$LAST/STATUS" 2>/dev/null; do sleep 60; done
log "$LAST finished: $(cat "$OUT/runs/$LAST/STATUS"); waiting for L1 decision file $GATE/{GO,STOP}"
until [ -e "$GATE/GO" ] || [ -e "$GATE/STOP" ]; do sleep 120; done
if [ -e "$GATE/STOP" ]; then log "STOP -> chain done"; exit 0; fi
log "GO -> s1, s2"
submit "t038-k$K-s1" 1 || true
submit "t038-k$K-s2" 2 || true
log "chain done"
