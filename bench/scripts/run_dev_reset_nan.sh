#!/bin/bash
# develop stack: random-action blow-up probe on native heightfield vs trimesh rough terrain
R=/workspace/bench/results_dev; PY=/workspace/IsaacLab-dev/.venv-lite/bin/python
COMMON="HOME=/root OMNI_KIT_ACCEPT_EULA=YES HEADLESS=1 AGILE_NEWTON_FLAT_TERRAIN=0 AGILE_NEWTON_NCONMAX=96 AGILE_NEWTON_SUBSTEPS=1 AGILE_NEWTON_IMPLICIT_ACTUATORS=1 AGILE_NEWTON_INTEGRATOR=implicitfast AGILE_NEWTON_DC_ENVELOPE=1 AGILE_NEWTON_LIMIT_SOLREF=0.02,1 AGILE_NEWTON_VEL_CLAMP=1 AGILE_NO_ASSIST=1"
for mode in hfield trimesh; do
  HF=1; [ $mode = trimesh ] && HF=0
  L=$R/reset_nan_random_${mode}_dev.log
  env $COMMON AGILE_NEWTON_HEIGHTFIELD=$HF $PY /workspace/bench/scripts/hfield_reset_nan_probe.py --envs 1024 --steps 300 --actions random --label $mode > $L 2>&1; echo "EXIT=$?" >> $L
done
echo ALLDONE > $R/reset_nan_random.done
