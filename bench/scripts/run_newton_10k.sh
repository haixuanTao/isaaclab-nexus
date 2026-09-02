#!/bin/bash
# Long Newton run on the physics stack: flat ground, implicit actuators, implicitfast,
# DC-motor envelope + generator braking + velocity band in-solver, MuJoCo-native joint
# limits, resting-state fallen dataset, velocity clamp kept ON as an impulse guard
# (engagement is logged as [clamp-stats]).  Usage: run_newton_10k.sh <iters> <num_envs>
set -u
ITERS="${1:-10000}"; N="${2:-4096}"
RES=/workspace/bench/results_newton
TAG="newton-physfix-guard_train${ITERS}_n${N}_$(date +%H%M%S)"
echo "=== [$(date +%T)] $TAG ==="
cd /workspace/WBC-AGILE-NEWTON
( env HOME=/root OMNI_KIT_ACCEPT_EULA=YES \
    AGILE_NEWTON_FLAT_TERRAIN=1 AGILE_NEWTON_SUBSTEPS=1 AGILE_NEWTON_IMPLICIT_ACTUATORS=1 AGILE_NEWTON_INTEGRATOR=implicitfast \
    AGILE_NEWTON_DC_ENVELOPE=1 AGILE_NEWTON_LIMIT_SOLREF=0.02,1 AGILE_NEWTON_VEL_CLAMP=1 AGILE_NEWTON_VEL_CLAMP_STATS=1 AGILE_NAN_WATCHDOG=1 \
    ./.venv/bin/python scripts/train.py --task HeightTracking-G1-v0 --num_envs "$N" \
    --headless --max_iterations "$ITERS" ) > "$RES/train_${TAG}.log" 2>&1
RC=$?
echo "    exit=$RC -> $RES/train_${TAG}.log"
sed -e 's/\x1b\[[0-9;]*m//g' "$RES/train_${TAG}.log" | grep -E "Learning iteration" | tail -1
sed -e 's/\x1b\[[0-9;]*m//g' "$RES/train_${TAG}.log" | grep -cE "contains NaN" | sed 's/^/    NaN hits: /'
