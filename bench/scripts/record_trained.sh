#!/bin/bash
# Record a video of the Newton-trained policy. Runs eval.py with the same
# AGILE_NEWTON_SUBSTEPS as training so the sim config matches.
# Usage: record_trained.sh [checkpoint] [num_envs] [seconds]
set -u
TREE=/workspace/WBC-AGILE-NEWTON
OUT=/workspace/bench/video
mkdir -p "$OUT"
CKPT="${1:-}"
N="${2:-16}"
SECS="${3:-12}"

if [ -z "$CKPT" ]; then   # newest checkpoint from the newest run
  CKPT=$(find "$TREE/logs/rsl_rl/height_tracking_g1" -name "model_*.pt" -printf "%T@ %p\n" \
         | sort -rn | head -1 | cut -d' ' -f2-)
fi
echo "checkpoint: $CKPT"

( cd "$TREE" && env HOME=/root OMNI_KIT_ACCEPT_EULA=YES AGILE_NEWTON_SUBSTEPS=1 PYGLET_HEADLESS=1 \
    ./.venv/bin/python scripts/eval.py --task HeightTracking-G1-v0 --num_envs "$N" \
    --viz newton --video --video_length_s "$SECS" --checkpoint "$CKPT" ) \
  > "$OUT/newton_trained_record.log" 2>&1
echo "exit=$?"

MP4=$(find "$TREE/logs" -name "*.mp4" -newermt "-20 minutes" -printf "%T@ %p\n" 2>/dev/null \
      | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$MP4" ]; then
  cp "$MP4" "$OUT/newton_trained.mp4"
  echo "-> $OUT/newton_trained.mp4 ($(du -h "$OUT/newton_trained.mp4" | cut -f1))"
else
  echo "!! no mp4 produced; see $OUT/newton_trained_record.log"
fi
