#!/bin/bash
# Newton throughput at a given substep count. PhysX baselines come from
# run_engine_compare.sh; this fills in the Newton side at a *stable* config.
# Usage: run_newton_substeps.sh <substeps> <task> <iters> <num_envs>...
set -u
SUB="${1:?substeps}"; TASK="${2:?task}"; ITERS="${3:?iters}"; shift 3
RES=/workspace/bench/results_newton
mkdir -p "$RES"
for N in "$@"; do
  TAG="newton-sub${SUB}_$(echo "$TASK" | tr -d '/')_n${N}"
  echo "=== [$(date +%T)] newton substeps=$SUB  $TASK  num_envs=$N  iters=$ITERS ==="
  /workspace/bench/scripts/power_sample.sh "$RES/power_${TAG}.csv" 100 &
  PS_PID=$!
  ( cd /workspace/WBC-AGILE-NEWTON && env HOME=/root OMNI_KIT_ACCEPT_EULA=YES \
      AGILE_NEWTON_SUBSTEPS="$SUB" ./.venv/bin/python scripts/train.py --task "$TASK" \
      --num_envs "$N" --headless --max_iterations "$ITERS" ) > "$RES/train_${TAG}.log" 2>&1
  echo "    exit=$?  -> $RES/train_${TAG}.log"
  kill $PS_PID 2>/dev/null; wait $PS_PID 2>/dev/null
done
echo "=== newton substeps=$SUB done $(date +%T) ==="
