#!/bin/bash
# Export + record a given checkpoint on ROUGH terrain (tiled heightfield). Usage: record_rough_checkpoint.sh <run_dir_name> <iter> <tag>
set -u
RUN="$1"; CK="$2"; TAG="$3"
cd /workspace/WBC-AGILE-NEWTON
PHYS="HOME=/root OMNI_KIT_ACCEPT_EULA=YES AGILE_NEWTON_FLAT_TERRAIN=0 AGILE_NEWTON_HEIGHTFIELD=1 AGILE_NEWTON_HEIGHTFIELD_TILE=8 AGILE_NEWTON_IMPLICIT_ACTUATORS=1 AGILE_NEWTON_INTEGRATOR=implicitfast AGILE_NEWTON_DC_ENVELOPE=1 AGILE_NEWTON_LIMIT_SOLREF=0.02,1 AGILE_NEWTON_VEL_CLAMP=1 AGILE_NEWTON_RESET_WARMSTART=1 AGILE_NO_ASSIST=1"
OUT=/workspace/bench/video/export_${TAG}_${CK}
env $PHYS ./.venv/bin/python scripts/export_policy.py --task HeightTracking-G1-v0 --load_run "$RUN" \
  --checkpoint "/workspace/WBC-AGILE-NEWTON/logs/rsl_rl/height_tracking_g1/$RUN/model_${CK}.pt" --resume True --output_dir "$OUT" > "$OUT.export.log" 2>&1
ls -la "$OUT/policy.pt" || { echo "[post] export failed"; exit 1; }
env $PHYS ./.venv/bin/python /workspace/bench/scripts/record_newton_direct.py --headless --num_envs 9 --seconds 15 \
  --checkpoint "$OUT/policy.pt" --out "/workspace/bench/video/${TAG}_${CK}.mp4" > "/workspace/bench/video/record_${TAG}_${CK}.log" 2>&1
grep -aE "^\[rec\]|^\[assist\]" "/workspace/bench/video/record_${TAG}_${CK}.log"
ls -la "/workspace/bench/video/${TAG}_${CK}.mp4"; echo RECDONE
