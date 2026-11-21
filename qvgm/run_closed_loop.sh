#!/usr/bin/env bash
set -uo pipefail
cd /workspace/StochRL/.claude/worktrees/qvgm-gate
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/StochRL/.claude/worktrees/qvgm-gate/third_party/RLinf:/workspace/StochRL/.claude/worktrees/qvgm-gate EMBODIED_PATH=/workspace/StochRL/.claude/worktrees/qvgm-gate/third_party/RLinf/examples/embodiment
export VK_ICD_FILENAMES=/workspace/StochRL/.claude/worktrees/qvgm-gate/repcritic/vulkan/nvidia_icd.json VK_DRIVER_FILES=/workspace/StochRL/.claude/worktrees/qvgm-gate/repcritic/vulkan/nvidia_icd.json
export TOKENIZERS_PARALLELISM=false HF_HOME=${HF_HOME:-/cache/huggingface} RC_POLICY_MB=8
OUT=/workspace/stochrl_t040/closed_loop
date +%s > $OUT/t_start
( while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 1; sleep 10; done ) > $OUT/gpu_trace.txt 2>/dev/null &
S=$!
/workspace/StochRL/.claude/worktrees/repcritic/.venv/bin/python -m qvgm.eval_qascent --out $OUT --budget_s 2200 --memprobe "$@"
RC=$?
kill $S; date +%s > $OUT/t_end; exit $RC
