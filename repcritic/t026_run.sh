#!/usr/bin/env bash
# T026 driver: restore test then main branches on GPU 1 (resumable; waits for free memory, retries on failure).
# usage: bash repcritic/t026_run.sh   (from repo root)
set -u
export CUDA_VISIBLE_DEVICES=1 RC_POLICY_MB=8
export RC_RLINF=${RC_RLINF:-/workspace/StochRL/.claude/worktrees/repcritic/third_party/RLinf}
LOG=/workspace/stochrl_t026/logs
mkdir -p $LOG
wait_free() {  # wait until GPU 1 has >= 13 GB free
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    [ "$used" -le 10500 ] && return
    echo "$(date +%T) waiting: GPU1 used ${used}MiB"; sleep 30
  done
}
run() {  # $1 = tag, rest = args
  tag=$1; shift
  for attempt in 1 2 3 4 5 6; do
    wait_free
    echo "$(date +%T) start $tag attempt $attempt"
    .venv/bin/python -m repcritic.t026_branch "$@" >> $LOG/$tag.log 2>&1 && { echo "$(date +%T) ok $tag"; return 0; }
    echo "$(date +%T) FAILED $tag attempt $attempt: $(grep -E 'Error' $LOG/$tag.log | tail -1)"; sleep 60
  done
  return 1
}
if [ ! -s /workspace/stochrl_t026/branch_results_restore_test.jsonl ]; then
  run restore_test --restore_test --E 48 || exit 1
fi
.venv/bin/python -m repcritic.t026_analyze --restore_test > $LOG/restore_test_analysis.txt
agree=$(.venv/bin/python -c "import json;print(json.load(open('/workspace/stochrl_t026/restore_test.json'))['mean_majority_agreement'])")
echo "restore agreement $agree"
.venv/bin/python -c "import sys; sys.exit(0 if $agree >= 0.9 else 1)" || { echo "RESTORE TEST FAILED"; exit 2; }
run main --E 48 || exit 1
echo "ALL DONE"
