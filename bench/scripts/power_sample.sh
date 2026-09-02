#!/bin/bash
# Sample GPU power/util/clocks at fixed interval. nvidia-smi works in unprivileged
# containers, unlike nsys --gpu-metrics (which needs elevated perf access).
# Usage: power_sample.sh <out.csv> [interval_ms]
OUT="${1:?usage: power_sample.sh <out.csv> [interval_ms]}"
IVL="${2:-100}"
echo "timestamp_s,power_w,util_gpu_pct,util_mem_pct,sm_clock_mhz,mem_used_mib,temp_c" > "$OUT"
T0=$(date +%s.%N)
while true; do
  LINE=$(nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory,clocks.sm,memory.used,temperature.gpu \
                    --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  [ -z "$LINE" ] && continue
  NOW=$(date +%s.%N)
  echo "$(echo "$NOW - $T0" | bc),$LINE" >> "$OUT"
  sleep "$(echo "scale=3; $IVL/1000" | bc)"
done
