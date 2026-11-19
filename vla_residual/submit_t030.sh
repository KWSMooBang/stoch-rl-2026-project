#!/usr/bin/env bash
# T030 submit chain, GPU 0 ONLY (human instruction 2026-11-20: budget 6.5 GPU-h, GPU 1/3 forbidden, order T028 -> T029 -> T030).
# Start gate: no t028-*/t029-* run is queued/running AND (>= T030_T029_N t029-* runs finished, refused ones not counted,
#   OR the release file $OUT/t030/GO exists). The gate must hold again after a 10 min settle (lets T029 resubmit).
# Then one run at a time: submit only when GPU 0 is free and none of ours is queued/running (harness picks the lowest
# free index -> GPU 0). run_t030.sh refuses any other GPU (exit 3) -> resubmitted as <run_id>-rN.
# Budget guard: skip a run if used (results.tsv t030-* gpu_h + T030_PRE_GPUH) + estimate > T030_BUDGET.
# usage: nohup setsid vla_residual/submit_t030.sh "t030-grpo-s1:grpo:1:1.3" ... > chain.log 2>&1 &
set -uo pipefail
WT=$(cd "$(dirname "$0")/.." && pwd)
cd "$WT"
OUT=/outputs/StochRL
BUDGET=${T030_BUDGET:-6.5}
PRE=${T030_PRE_GPUH:-0.1}
MAXPAR=${T030_MAXPAR:-1}
T029_N=${T030_T029_N:-2}

log() { echo "[$(date -Is)] $*"; }
free_0() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '$1==0 && $2<1000 {f=1} END {print f+0}'; }
count() {  # count <run glob prefix> <status regex>
  local n=0 s
  for s in "$OUT"/runs/$1-*/STATUS; do
    [ -f "$s" ] || continue
    grep -qE "^($2)$" "$s" && n=$((n+1))
  done
  echo $n
}
t029_finished() {  # finished t029 runs that actually trained (not refused by a GPU guard)
  local n=0 d
  for d in "$OUT"/runs/t029-*; do
    [ -f "$d/STATUS" ] || continue
    grep -qE '^(done|failed|timeout)$' "$d/STATUS" || continue
    grep -q REFUSE "$d/stdout.log" 2>/dev/null && continue
    n=$((n+1))
  done
  echo $n
}
gate() {
  [ "$(count t028 'queued|running')" -eq 0 ] && [ "$(count t029 'queued|running')" -eq 0 ] && \
    { [ -e "$OUT/t030/GO" ] || [ "$(t029_finished)" -ge "$T029_N" ]; }
}
used_gpuh() { awk -F'\t' -v pre="$PRE" '$1 ~ /^t030-/ {s+=$5} END {printf "%.2f", s+pre}' "$OUT/results.tsv" 2>/dev/null; }

log "waiting for T028/T029 (need $T029_N finished t029 runs or $OUT/t030/GO)"
while true; do
  until gate; do sleep 120; done
  log "gate open (t029 finished $(t029_finished)); settle 10 min"
  sleep 600
  gate && break
done
log "start T030 chain"

for spec in "$@"; do
  IFS=: read -r BASE COND SEED HOURS <<< "$spec"
  n=0
  while true; do
    until [ "$(count t030 'queued|running')" -lt "$MAXPAR" ] && [ "$(count t030 queued)" -eq 0 ] && \
          [ "$(count t029 'queued|running')" -eq 0 ] && [ "$(free_0)" -eq 1 ]; do sleep 60; done
    U=$(used_gpuh)
    if awk -v u="$U" -v h="$HOURS" -v b="$BUDGET" 'BEGIN{exit !(u+h>b)}'; then
      log "SKIP $BASE: budget used $U + est $HOURS > $BUDGET"; break
    fi
    ID=$BASE; [ $n -gt 0 ] && ID=$BASE-r$n
    log "submit $ID (used $U GPU-h, est $HOURS)"
    RUN_HOURS=$(awk -v h="$HOURS" 'BEGIN{printf "%.2f", h*1.4+0.2}') /harness/launch.sh "$ID" 1 vla_residual/run_t030.sh "$COND" "$SEED"
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
