#!/bin/bash
# Throughput sweep for WBC-AGILE (G1 29-DOF full-body stand-up/height-tracking).
# One iteration = 24 rollout steps x num_envs, then 5 epochs x 4 minibatches.
set -u
cd /workspace/WBC-AGILE
TASK="${1:?task}"; ITERS="${2:?iters}"; shift 2
RES=/workspace/bench/results
mkdir -p "$RES"
for N in "$@"; do
  TAG="$(echo "$TASK" | tr -d '/')_n${N}"
  echo "=== [$(date +%T)] $TASK num_envs=$N iters=$ITERS ==="
  /workspace/bench/scripts/power_sample.sh "$RES/power_${TAG}.csv" 100 &
  PS_PID=$!
  env HOME=/root OMNI_KIT_ACCEPT_EULA=YES \
    uv run scripts/train.py --task "$TASK" --num_envs "$N" --headless \
      --max_iterations "$ITERS" > "$RES/train_${TAG}.log" 2>&1
  RC=$?
  kill $PS_PID 2>/dev/null; wait $PS_PID 2>/dev/null
  echo "    exit=$RC  -> $RES/train_${TAG}.log"
done
echo "=== sweep done $(date +%T) ==="
