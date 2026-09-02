#!/usr/bin/env python3
"""Does the Newton MODEL carry its own joint drives (target_ke/kd, actuators)
on top of AGILE's explicit DCMotor torques? Double actuation would produce
torque the Isaac Lab side never sees."""
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, torch, warp as wp
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
cfg = parse_env_cfg(args_cli.task, num_envs=2); env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped
env.reset()
from isaaclab_newton.physics import NewtonManager
m = NewtonManager.get_model()
def arr(name):
    a = getattr(m, name, None)
    if a is None: return None
    try: return wp.to_torch(a) if isinstance(a, wp.array) else torch.as_tensor(a)
    except Exception as e: return f"<{e}>"
print("\n[drv] Newton model joint drive / actuator fields:")
for f in ["joint_target_ke","joint_target_kd","joint_target_mode","joint_armature","joint_friction",
          "joint_effort_limit","joint_velocity_limit","joint_limit_ke","joint_limit_kd"]:
    a = arr(f)
    if isinstance(a, torch.Tensor) and a.numel():
        a = a.float()
        print(f"[drv]   {f:22s} n={a.numel():5d}  min={a.min():10.4f} max={a.max():10.4f} nonzero={(a!=0).sum().item()}")
    else:
        print(f"[drv]   {f:22s} {a}")
for f in ["joint_act","joint_f","joint_target_pos","joint_target_vel"]:
    c = getattr(NewtonManager.get_control(), f, None) if hasattr(NewtonManager, "get_control") else None
    if isinstance(c, wp.array):
        t = wp.to_torch(c).float()
        print(f"[drv]   control.{f:18s} n={t.numel():5d}  absmax={t.abs().max():10.4f} nonzero={(t!=0).sum().item()}")
# MuJoCo-side: does the solver have actuators?
for f in ["actuator_count","nu","mjc_actuator_count"]:
    v = getattr(m, f, None)
    if v is not None: print(f"[drv]   model.{f} = {v}")
env.close(); simulation_app.close()
