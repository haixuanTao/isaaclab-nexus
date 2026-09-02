#!/bin/bash
# After a run: pick the best checkpoint by mean reward, export TorchScript, record harness-free.
# Usage: post_train_record.sh <train_log> <run_dir_name> <video_tag>
set -u
LOG="$1"; RUN="$2"; TAG="$3"
cd /workspace/WBC-AGILE-NEWTON
PHYS="HOME=/root OMNI_KIT_ACCEPT_EULA=YES AGILE_NEWTON_FLAT_TERRAIN=1 AGILE_NEWTON_IMPLICIT_ACTUATORS=1 AGILE_NEWTON_INTEGRATOR=implicitfast AGILE_NEWTON_DC_ENVELOPE=1 AGILE_NEWTON_LIMIT_SOLREF=0.02,1 AGILE_NEWTON_VEL_CLAMP=1"
# mean reward per iteration -> best checkpoint among the saved ones
BEST=$(sed -e 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -aE "Mean reward:" | awk '{print NR-1, $3}' | python3 -c '
import sys
rows=[(int(a),float(b)) for a,b in (l.split() for l in sys.stdin)]
import glob,os,re
ck=sorted(int(re.search(r"model_(\d+)\.pt",p).group(1)) for p in glob.glob("logs/rsl_rl/height_tracking_g1/'"$RUN"'/model_*.pt"))
# score each checkpoint by the mean reward over the 20 iterations before it
def score(c):
    w=[r for i,r in rows if c-20<=i<=c]; return sum(w)/len(w) if w else -1e9
best=max(ck,key=score); print(best, round(score(best),1))')
CK=$(echo $BEST | cut -d' ' -f1); SC=$(echo $BEST | cut -d' ' -f2)
echo "[post] best checkpoint model_${CK}.pt (mean reward over its last 20 iters: $SC)"
OUT=/workspace/bench/video/export_${TAG}_${CK}
env $PHYS ./.venv/bin/python scripts/export_policy.py --task HeightTracking-G1-v0 --load_run "$RUN" \
  --checkpoint "/workspace/WBC-AGILE-NEWTON/logs/rsl_rl/height_tracking_g1/$RUN/model_${CK}.pt" --resume True --output_dir "$OUT" > "$OUT.export.log" 2>&1
ls -la "$OUT/policy.pt" || { echo "[post] export failed"; exit 1; }
env $PHYS ./.venv/bin/python /workspace/bench/scripts/record_newton_direct.py --headless --num_envs 9 --seconds 15 \
  --checkpoint "$OUT/policy.pt" --out "/workspace/bench/video/${TAG}_${CK}.mp4" > "/workspace/bench/video/record_${TAG}_${CK}.log" 2>&1
grep -aE "^\[rec\]|^\[assist\]" "/workspace/bench/video/record_${TAG}_${CK}.log"
ls -la "/workspace/bench/video/${TAG}_${CK}.mp4"
