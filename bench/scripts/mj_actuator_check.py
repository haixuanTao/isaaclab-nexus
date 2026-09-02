#!/usr/bin/env python3
"""What does the live MuJoCo model actually contain: actuator gain/bias, passive
dof damping/stiffness, limits? Anything nonzero here is a force Isaac Lab's
explicit actuator does not know about."""
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, torch, numpy as np, warp as wp
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
cfg = parse_env_cfg(args_cli.task, num_envs=2); env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped
env.reset()
import gc
solver = None
for o in gc.get_objects():
    try:
        if type(o).__name__ == "SolverMuJoCo": solver = o; break
    except Exception: continue
mj = solver.mj_model
def a(x): return np.asarray(x)
print(f"\n[mj] nu (actuators)={mj.nu}  nv (dofs)={mj.nv}  nq={mj.nq}  njnt={mj.njnt}")
def summ(name, arr):
    arr = a(arr); nz = np.count_nonzero(np.abs(arr) > 1e-9)
    print(f"[mj] {name:22s} shape={arr.shape}  nonzero={nz}  min={arr.min():+.4g}  max={arr.max():+.4g}")
for f in ("actuator_gainprm", "actuator_biasprm", "actuator_gear", "actuator_ctrlrange", "actuator_forcerange",
          "actuator_dyntype", "actuator_gaintype", "actuator_biastype", "actuator_trntype"):
    if hasattr(mj, f): summ(f, getattr(mj, f))
for f in ("dof_damping", "dof_armature", "dof_frictionloss", "jnt_stiffness", "jnt_limited", "jnt_range", "jnt_solref", "jnt_solimp"):
    if hasattr(mj, f): summ(f, getattr(mj, f))
print(f"[mj] opt.timestep={mj.opt.timestep}  opt.integrator={mj.opt.integrator}  opt.solver={mj.opt.solver}  iterations={mj.opt.iterations}  "
      f"opt.gravity={a(mj.opt.gravity)}  opt.cone={mj.opt.cone}  opt.impratio={mj.opt.impratio}")
# first few actuators in detail
for i in range(min(6, mj.nu)):
    print(f"[mj] act[{i}] gain={a(mj.actuator_gainprm)[i,:3]} bias={a(mj.actuator_biasprm)[i,:3]} gear={a(mj.actuator_gear)[i,0]:.3g} "
          f"trn={a(mj.actuator_trnid)[i]} ctrlrange={a(mj.actuator_ctrlrange)[i]}")
env.close(); simulation_app.close()
