#!/bin/bash
# nsys trace of a steady-state WBC-AGILE training window.
# CPU sampling is disabled: this container has perf_event_paranoid=4 and
# perf_event_open is unavailable, so --sample/--cpuctxsw would fail. CUPTI-based
# CUDA tracing does not need perf and works fine.
# Usage: run_nsys.sh <task> <num_envs> <delay_s> <duration_s> [tag]
set -u
cd /workspace/WBC-AGILE
TASK="${1:?task}"; N="${2:?num_envs}"; DELAY="${3:-180}"; DUR="${4:-60}"
TAG="${5:-${TASK}_n${N}}"
OUT=/workspace/bench/traces/agile_${TAG}
mkdir -p /workspace/bench/traces

/workspace/bench/scripts/power_sample.sh "/workspace/bench/results/power_nsys_${TAG}.csv" 50 &
PS_PID=$!

nsys profile \
  --output="$OUT" --force-overwrite=true \
  --trace=cuda,osrt,nvtx \
  --sample=none --cpuctxsw=none \
  --cuda-memory-usage=false \
  --delay="$DELAY" --duration="$DUR" \
  --kill=sigkill \
  uv run scripts/train.py --task "$TASK" --num_envs "$N" --headless \
    --max_iterations 100000 \
  > "/workspace/bench/results/nsys_${TAG}.log" 2>&1
RC=$?
kill $PS_PID 2>/dev/null
echo "nsys exit=$RC -> ${OUT}.nsys-rep"
