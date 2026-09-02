#!/bin/bash
# 1000-iteration Newton training with implicit (solver-side PD) actuators.
# Usage: run_newton_implicit_train.sh <iters> <num_envs> [substeps]
set -u
ITERS="${1:-1000}"; N="${2:-4096}"; SUB="${3:-1}"; INT="${4:-implicitfast}"
RES=/workspace/bench/results_newton
TAG="newton-implicit-${INT}-sub${SUB}_train${ITERS}_n${N}_$(date +%H%M%S)"
/workspace/bench/scripts/power_sample.sh "$RES/power_${TAG}.csv" 500 &
PS=$!
echo "=== [$(date +%T)] $TAG ==="
( env HOME=/root OMNI_KIT_ACCEPT_EULA=YES \
    AGILE_NEWTON_SUBSTEPS="$SUB" AGILE_NEWTON_IMPLICIT_ACTUATORS=1 AGILE_NEWTON_INTEGRATOR="$INT" AGILE_NAN_WATCHDOG=1 \
    ./.venv/bin/python scripts/train.py --task HeightTracking-G1-v0 --num_envs "$N" \
    --headless --max_iterations "$ITERS" ) > "$RES/train_${TAG}.log" 2>&1
RC=$?
kill $PS 2>/dev/null; wait $PS 2>/dev/null
echo "    exit=$RC -> $RES/train_${TAG}.log"
sed -e 's/\x1b\[[0-9;]*m//g' "$RES/train_${TAG}.log" | grep -E "Exact experiment name|Learning iteration" | tail -2
sed -e 's/\x1b\[[0-9;]*m//g' "$RES/train_${TAG}.log" | grep -cE "contains NaN" | sed 's/^/    NaN hits: /'
