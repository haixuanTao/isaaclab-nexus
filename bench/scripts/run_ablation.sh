#!/bin/bash
# Ablation ladder at fixed num_envs. All rungs go through train_ablate.py so the
# harness itself is held constant; only AGILE_ABLATE changes.
set -u
cd /workspace/WBC-AGILE
N="${1:-4096}"; ITERS="${2:-20}"
RES=/workspace/bench/results
for MODE in none nophysics nophysics_noreadback; do
  TAG="ablate_${MODE}_n${N}"
  echo "=== [$(date +%T)] AGILE_ABLATE=$MODE num_envs=$N ==="
  /workspace/bench/scripts/power_sample.sh "$RES/power_${TAG}.csv" 100 &
  PS=$!
  env HOME=/root OMNI_KIT_ACCEPT_EULA=YES AGILE_ABLATE="$MODE" AGILE_NVTX=0 \
    ./.venv/bin/python /workspace/bench/scripts/train_ablate.py \
      --task HeightTracking-G1-v0 --num_envs "$N" --headless --max_iterations "$ITERS" \
    > "$RES/${TAG}.log" 2>&1
  echo "    exit=$? -> $RES/${TAG}.log"
  kill $PS 2>/dev/null; wait $PS 2>/dev/null
done
echo "=== ablation ladder done $(date +%T) ==="
