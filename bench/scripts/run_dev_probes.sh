#!/bin/bash
# Physics probe set on the Isaac Lab develop + Newton 1.5.1 stack.
# Usage: run_dev_probes.sh <tag> [extra env assignments...]
set -u
TAG="$1"; shift
PY=/workspace/IsaacLab-dev/.venv-lite/bin/python
TREE=/workspace/WBC-AGILE-NEWTON-DEV
RES=/workspace/bench/results_dev; mkdir -p $RES
BASE="HOME=/root OMNI_KIT_ACCEPT_EULA=YES AGILE_NEWTON_FLAT_TERRAIN=1 AGILE_NEWTON_IMPLICIT_ACTUATORS=1 AGILE_NEWTON_INTEGRATOR=implicitfast AGILE_NEWTON_VEL_CLAMP=0 $*"
cd $TREE
run() { name=$1; shift; echo "--- $name"; env $BASE $PY "$@" > "$RES/${TAG}_$name.log" 2>&1; echo "    exit=$?"; }
run drop          /workspace/bench/scripts/drop_impact_compare.py --headless --label $TAG
run tilt          /workspace/bench/scripts/moment_sign_check.py --headless --label $TAG
run tilt_rigid    /workspace/bench/scripts/moment_sign_check.py --headless --label ${TAG}_rigid --rigid
run limit_push    /workspace/bench/scripts/limit_push_probe.py --headless --label $TAG
run blowup        /workspace/bench/scripts/blowup_probe.py --headless --label $TAG
run reset_stress  /workspace/bench/scripts/reset_stress_probe.py --headless --envs 2048 --steps 1500 --reset_write rest
for f in $RES/${TAG}_*.log; do echo "######## $(basename $f)"; grep -aE "RESULT|SURVIVED|NON-FINITE|OK|WRONG SIGN|overshoot|cosine|Traceback|Error:" "$f" | grep -avE "omni|carb" | tail -4; done
