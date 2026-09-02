#!/bin/bash
# PhysX vs Newton throughput comparison.
#
# Both trees are the same WBC-AGILE commit; the Newton tree differs only in
# `sim.physics = NewtonCfg()` (plus the two PhysX-only knobs it drops and the
# backend-agnostic effort-limit lookup those rewards need). Same task, same
# num_envs, same iteration count, same interpreter invocation, back to back on
# an otherwise idle GPU.
#
# Usage: run_engine_compare.sh <task> <iters> <num_envs>...
set -u
TASK="${1:?task}"; ITERS="${2:?iters}"; shift 2
RES=/workspace/bench/results_newton
mkdir -p "$RES"

for N in "$@"; do
  for ENGINE in physx newton; do
    case "$ENGINE" in
      physx)  TREE=/workspace/WBC-AGILE ;;
      newton) TREE=/workspace/WBC-AGILE-NEWTON ;;
    esac
    TAG="${ENGINE}_$(echo "$TASK" | tr -d '/')_n${N}"
    echo "=== [$(date +%T)] $ENGINE  $TASK  num_envs=$N  iters=$ITERS ==="
    /workspace/bench/scripts/power_sample.sh "$RES/power_${TAG}.csv" 100 &
    PS_PID=$!
    ( cd "$TREE" && env HOME=/root OMNI_KIT_ACCEPT_EULA=YES \
        ./.venv/bin/python scripts/train.py --task "$TASK" --num_envs "$N" \
        --headless --max_iterations "$ITERS" ) > "$RES/train_${TAG}.log" 2>&1
    RC=$?
    kill $PS_PID 2>/dev/null; wait $PS_PID 2>/dev/null
    echo "    exit=$RC  -> $RES/train_${TAG}.log"
  done
done
echo "=== compare done $(date +%T) ==="
