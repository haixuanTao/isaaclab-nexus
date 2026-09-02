#!/usr/bin/env python3
"""Does the armature Isaac Lab writes (0.02) actually reach MJWarp's dof_armature?
Compares newton model.joint_armature vs the live mujoco-warp model's dof_armature."""
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, torch, warp as wp, numpy as np
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
cfg = parse_env_cfg(args_cli.task, num_envs=2); env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; robot = u.scene["robot"]
env.reset()
from isaaclab_newton.physics import NewtonManager
m = NewtonManager.get_model()
ja = wp.to_torch(m.joint_armature).float().cpu().numpy()
print(f"[arm] newton model.joint_armature: n={ja.size} unique={np.unique(np.round(ja,4))[:8]}")
# find the live solver and its mujoco-warp model
solver = None
for attr in dir(NewtonManager):
    v = getattr(NewtonManager, attr, None)
    if v is not None and hasattr(v, "mjw_model"): solver = v; break
if solver is None:
    import gc
    for o in gc.get_objects():
        try:
            if hasattr(o, "mjw_model") and hasattr(o, "mj_model"): solver = o; break
        except Exception: continue
print(f"[arm] solver object: {type(solver).__name__ if solver else None}")
if solver is not None:
    da = solver.mjw_model.dof_armature
    da = wp.to_torch(da).float().cpu().numpy() if isinstance(da, wp.array) else np.asarray(da)
    print(f"[arm] mjw_model.dof_armature: shape={da.shape} unique={np.unique(np.round(da.ravel(),4))[:8]}")
    print(f"[arm] mj_model.dof_armature (cpu mirror): unique={np.unique(np.round(np.asarray(solver.mj_model.dof_armature),4))[:8]}")
    ia = robot.data.joint_armature; ia = ia.torch if hasattr(ia, "torch") else ia
    print(f"[arm] isaaclab robot.data.joint_armature: unique={np.unique(np.round(ia[0].cpu().numpy(),4))[:8]}")
    print(f"[arm] VERDICT: {'ARMATURE REACHES MJWARP' if np.any(np.abs(da.ravel()-0.02)<1e-4) else 'ARMATURE NOT APPLIED IN MJWARP (solver runs with different values)'}")
env.close(); simulation_app.close()
