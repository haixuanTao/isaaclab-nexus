#!/bin/bash
# Nexus-backend throughput sweep on AGILE's HeightTracking-G1-v0, same shape as
# bench/results/train_HeightTracking-G1-v0_n*.log (PhysX baseline).
cd /workspace/WBC-AGILE || exit 1
OUT=/workspace/bench/results
for N in 1024 2048 4096; do
  echo "=== nexus n=$N ==="
  OMNI_KIT_ACCEPT_EULA=YES HOME=/root .venv/bin/python \
    /workspace/bench/nexus_port/train_nexus.py "$N" 10 \
    > "$OUT/train_nexus_HeightTracking-G1-v0_n${N}.log" 2>&1
  echo "n=$N exit $?"
  grep -E "\[nexus\]" "$OUT/train_nexus_HeightTracking-G1-v0_n${N}.log" | tail -3
done
echo NEXUS_SWEEP_DONE
