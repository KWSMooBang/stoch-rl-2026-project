#!/usr/bin/env bash
# T024 submit chain (same rules as submit_chain.sh): submit only when GPU 0 or 1 is free and nothing of ours is
# queued; run.sh refuses any other GPU (exit 3) -> resubmitted as <run_id>-rN. GPU 3 never used.
# Budget guard: skip a run if used (results.tsv t024-* gpu_h + T024_PRE_GPUH) + estimate > T024_BUDGET.
# usage: nohup setsid vla_residual/submit_t024.sh "t024-vla-b10-s0:vla:1.0:0:1.3" ... > chain.log 2>&1 &
set -uo pipefail
WT=$(cd "$(dirname "$0")/.." && pwd)
cd "$WT"
OUT=/outputs/StochRL
BUDGET=${T024_BUDGET:-6.5}
PRE=${T024_PRE_GPUH:-0}
MAXPAR=${T024_MAXPAR:-2}

log() { echo "[$(date -Is)] $*"; }
free_01() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '($1==0||$1==1) && $2<1000 {f=1} END {print f+0}'; }
count() {  # count <status regex>
  local n=0 s
  for s in "$OUT"/runs/t024-*/STATUS; do
    [ -f "$s" ] || continue
    grep -qE "^($1)$" "$s" && n=$((n+1))
  done
  echo $n
}
used_gpuh() { awk -F'\t' -v pre="$PRE" '$1 ~ /^t024-/ {s+=$5} END {printf "%.2f", s+pre}' "$OUT/results.tsv" 2>/dev/null; }

for spec in "$@"; do
  IFS=: read -r BASE POT BETA SEED HOURS <<< "$spec"
  n=0
  while true; do
    until [ "$(count 'queued|running')" -lt "$MAXPAR" ] && [ "$(count queued)" -eq 0 ] && [ "$(free_01)" -eq 1 ]; do sleep 60; done
    U=$(used_gpuh)
    if awk -v u="$U" -v h="$HOURS" -v b="$BUDGET" 'BEGIN{exit !(u+h>b)}'; then
      log "SKIP $BASE: budget used $U + est $HOURS > $BUDGET"; break
    fi
    ID=$BASE; [ $n -gt 0 ] && ID=$BASE-r$n
    log "submit $ID (used $U GPU-h, est $HOURS)"
    RUN_HOURS=$(awk -v h="$HOURS" 'BEGIN{printf "%.2f", h*1.4+0.2}') /harness/launch.sh "$ID" 1 vla_residual/run_t024.sh "$POT" "$BETA" "$SEED"
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
