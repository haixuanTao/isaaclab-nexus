#!/bin/bash
# Develop-stack (Isaac Lab develop / Newton 1.5.1) training smokes: flat plane, then native heightfield.
cd /workspace/WBC-AGILE-NEWTON-DEV
RES=/workspace/bench/results_dev; PY=/workspace/IsaacLab-dev/.venv-lite/bin/python
COMMON="AGILE_NEWTON_RESET_WARMSTART=1 AGILE_NEWTON_NCONMAX=${NCONMAX:-96} HOME=/root OMNI_KIT_ACCEPT_EULA=YES HEADLESS=1 AGILE_NEWTON_SUBSTEPS=1 AGILE_NEWTON_IMPLICIT_ACTUATORS=1 AGILE_NEWTON_INTEGRATOR=implicitfast AGILE_NEWTON_DC_ENVELOPE=1 AGILE_NEWTON_LIMIT_SOLREF=0.02,1 AGILE_NEWTON_VEL_CLAMP=1 AGILE_NEWTON_VEL_CLAMP_STATS=1 AGILE_NAN_WATCHDOG=1 AGILE_NEWTON_NAN_QUARANTINE=1 AGILE_CLIP_ACTIONS=10 AGILE_NO_ASSIST=1"
N=${N:-1024}; IT=${IT:-20}
for mode in ${MODES:-flat hfield}; do
  if [ $mode = flat ]; then T="AGILE_NEWTON_FLAT_TERRAIN=1 AGILE_NEWTON_HEIGHTFIELD=0"; else T="AGILE_NEWTON_FLAT_TERRAIN=0 AGILE_NEWTON_HEIGHTFIELD=1"; fi
  L=$RES/train_dev_${mode}_smoke.log
  env $COMMON $T $PY scripts/train.py --task HeightTracking-G1-v0 --num_envs $N --max_iterations $IT > $L 2>&1; echo "EXIT=$?" >> $L
done
