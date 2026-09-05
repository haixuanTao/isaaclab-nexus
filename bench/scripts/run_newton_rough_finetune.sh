#!/bin/bash
# Fine-tune a checkpoint on ROUGH terrain (tiled heightfield collider), harness off, same physics stack.
# Usage: run_newton_rough_finetune.sh <load_run> <checkpoint.pt> <total_iters> <num_envs>
set -u
LOAD_RUN="$1"; CKPT="$2"; ITERS="${3:-10000}"; N="${4:-4096}"
RES=/workspace/bench/results_newton
TAG="newton-rough-hfield-ft_from${CKPT%.pt}_train${ITERS}_n${N}_$(date +%H%M%S)"
echo "=== [$(date +%T)] $TAG ==="
cd /workspace/WBC-AGILE-NEWTON
( env HOME=/root OMNI_KIT_ACCEPT_EULA=YES \
    AGILE_NEWTON_FLAT_TERRAIN=0 AGILE_NEWTON_HEIGHTFIELD=1 AGILE_NEWTON_HEIGHTFIELD_TILE=8 AGILE_NEWTON_SUBSTEPS=1 AGILE_NEWTON_IMPLICIT_ACTUATORS=1 AGILE_NEWTON_INTEGRATOR=implicitfast \
    AGILE_NEWTON_DC_ENVELOPE=1 AGILE_NEWTON_LIMIT_SOLREF=0.02,1 AGILE_NEWTON_VEL_CLAMP=1 AGILE_NEWTON_VEL_CLAMP_STATS=1 \
    AGILE_NAN_WATCHDOG=1 AGILE_NEWTON_NAN_QUARANTINE=1 AGILE_NEWTON_RESET_WARMSTART=1 AGILE_CLIP_ACTIONS=10 AGILE_NO_ASSIST=1 \
    ./.venv/bin/python scripts/train.py --task HeightTracking-G1-v0 --num_envs "$N" --headless --max_iterations "$ITERS" \
    --resume True --load_run "$LOAD_RUN" --checkpoint "$CKPT" ) > "$RES/train_${TAG}.log" 2>&1
RC=$?; echo "    exit=$RC -> $RES/train_${TAG}.log"
sed -e 's/\x1b\[[0-9;]*m//g' "$RES/train_${TAG}.log" | grep -E "Learning iteration" | tail -1
