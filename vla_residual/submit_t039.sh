#!/usr/bin/env bash
# T039 submit chain, GPU 0 ONLY (human instruction 2026-11-21: cap 5.0 GPU-h incl. validation, GPU 1/3 forbidden).
# Stage 1: submit s0 only. Then wait until s0 has finished AND L1 has written /outputs/StochRL/t039/GO or .../STOP
#          (pre-registered keep: once >= 0.68 and end >= 0.58 on s0's metrics.json).
#          GO -> stage 2 (s1, s2 sequentially); STOP -> chain ends.
# A run is submitted only when GPU 0 is free (< 1000 MiB) and no t036-t039 run is queued/running (the harness picks
# the lowest free index -> GPU 0). run_t039.sh refuses any other GPU (exit 3) -> resubmitted as <run_id>-rN.
# Budget guard: skip a run if used (results.tsv t039-* gpu_h + T039_PRE_GPUH) + estimate > T039_BUDGET (s2 first).
# usage: T039_PRE_GPUH=<validation+smoke GPU-h> nohup setsid vla_residual/submit_t039.sh > /outputs/StochRL/t039/chain.log 2>&1 &
set -uo pipefail
WT=$(cd "$(dirname "$0")/.." && pwd)
cd "$WT"
OUT=/outputs/StochRL
BUDGET=${T039_BUDGET:-5.0}
PRE=${T039_PRE_GPUH:?validation + smoke GPU-h spent outside the harness}
EST=${T039_EST_GPUH:-1.35}
GATE=$OUT/t039

log() { echo "[$(date -Is)] $*"; }
free_0() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '$1==0 && $2<1000 {f=1} END {print f+0}'; }
busy() {  # StochRL runs of T036-T039 queued/running
  local n=0 s
  for s in "$OUT"/runs/t03[6-9]-*/STATUS; do
    [ -f "$s" ] || continue
    grep -qE '^(queued|running)$' "$s" && n=$((n+1))
  done
  echo $n
}
used_gpuh() { awk -F'\t' -v pre="$PRE" '$1 ~ /^t039-/ {s+=$5} END {printf "%.2f", s+pre}' "$OUT/results.tsv" 2>/dev/null; }

submit() {  # submit <base id> <seed>; echoes the final run id (or SKIP)
  local BASE=$1 SEED=$2 n=0 ID U G
  while true; do
    until [ "$(busy)" -eq 0 ] && [ "$(free_0)" -eq 1 ]; do sleep 60; done
    sleep 20  # let the harness append the previous run's results.tsv row
    U=$(used_gpuh)
    if awk -v u="$U" -v h="$EST" -v b="$BUDGET" 'BEGIN{exit !(u+h>b)}'; then
      log "SKIP $BASE: budget used $U + est $EST > $BUDGET" >&2; echo SKIP; return
    fi
    ID=$BASE; [ $n -gt 0 ] && ID=$BASE-r$n
    log "submit $ID (used $U GPU-h, est $EST)" >&2
    RUN_HOURS=$(awk -v h="$EST" 'BEGIN{printf "%.2f", h*1.4+0.2}') /harness/launch.sh "$ID" 1 vla_residual/run_t039.sh "$SEED" >&2
    while [ "$(cat "$OUT/runs/$ID/STATUS" 2>/dev/null)" = queued ]; do sleep 30; done
    sleep 90
    G=$(cat "$OUT/runs/$ID/gpus" 2>/dev/null)
    if grep -q "REFUSE" "$OUT/runs/$ID/stdout.log" 2>/dev/null; then
      log "$ID refused on GPU $G -> resubmit" >&2; n=$((n+1)); [ $n -gt 20 ] && { log "give up $BASE" >&2; echo SKIP; return; }
      sleep 120; continue
    fi
    log "$ID running on GPU $G" >&2
    echo "$ID"; return
  done
}

log "start T039 chain (budget $BUDGET, pre $PRE, est $EST/run)"
S0=$(submit t039-noise-s0 0)
[ "$S0" = SKIP ] && { log "s0 skipped -> chain done"; exit 0; }
until grep -qE '^(done|failed|timeout|killed|error)' "$OUT/runs/$S0/STATUS" 2>/dev/null; do sleep 120; done
log "$S0 finished: $(cat "$OUT/runs/$S0/STATUS"); waiting for $GATE/GO or $GATE/STOP (written by L1)"
until [ -e "$GATE/GO" ] || [ -e "$GATE/STOP" ]; do sleep 120; done
if [ -e "$GATE/STOP" ]; then log "STOP -> chain done (s1, s2 not submitted)"; exit 0; fi
log "GO -> s1, s2"
for SEED in 1 2; do
  R=$(submit t039-noise-s$SEED $SEED)
  [ "$R" = SKIP ] && continue
done
log "chain done"
