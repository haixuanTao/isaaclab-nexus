#!/bin/bash
# Record short rollouts of the same task under each physics config, same seed,
# same (freshly initialized, untrained) policy, so the clips are comparable.
# Produces one mp4 per config under /workspace/bench/video/.
set -u
TASK="${TASK:-HeightTracking-G1-v0}"
N="${N:-16}"
LEN="${LEN:-300}"
OUT=/workspace/bench/video
mkdir -p "$OUT"

record() {  # name  tree  extra_env
  local NAME="$1" TREE="$2"; shift 2
  local LOGROOT="$TREE/logs/rsl_rl/height_tracking_g1"
  echo "=== [$(date +%T)] recording $NAME ==="
  local BEFORE
  BEFORE=$(ls -1 "$LOGROOT" 2>/dev/null | wc -l)
  ( cd "$TREE" && env HOME=/root OMNI_KIT_ACCEPT_EULA=YES "$@" \
      ./.venv/bin/python scripts/train.py --task "$TASK" --num_envs "$N" \
      --headless --video --video_length "$LEN" --video_interval_iter 1 \
      --max_iterations 2 ) > "$OUT/${NAME}_record.log" 2>&1
  echo "    exit=$?"
  # newest mp4 under that tree's logs
  local MP4
  MP4=$(find "$LOGROOT" -name "*.mp4" -newermt "-10 minutes" -printf "%T@ %p\n" 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2-)
  if [ -n "$MP4" ]; then
    cp "$MP4" "$OUT/${NAME}.mp4"
    echo "    -> $OUT/${NAME}.mp4  ($(du -h "$OUT/${NAME}.mp4" | cut -f1))"
  else
    echo "    !! no mp4 produced for $NAME (see $OUT/${NAME}_record.log)"
  fi
}

record physx        /workspace/WBC-AGILE
record newton_sub1  /workspace/WBC-AGILE-NEWTON  AGILE_NEWTON_SUBSTEPS=1
record newton_sub4  /workspace/WBC-AGILE-NEWTON  AGILE_NEWTON_SUBSTEPS=4
echo "=== recording done $(date +%T) ==="
