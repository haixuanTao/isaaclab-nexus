#!/usr/bin/env python3
"""Read joints BY NAME immediately after a dataset-loaded reset, BEFORE any sim
step (so the solver cannot project them back inside limits). Also report, per
named joint, the absmax seen -- which reveals which cache column it received."""
import argparse, sys, importlib
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--label", type=str, default="engine")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, torch
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
def tt(x): return x.torch if hasattr(x, "torch") else x
cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs); cfg.seed = 42
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; robot = u.scene["robot"]
spec = gym.spec(args_cli.task)
mn, cn = spec.kwargs["rsl_rl_cfg_entry_point"].split(":"); agent_cfg = getattr(importlib.import_module(mn), cn)()
mn, fn = spec.kwargs["pre_learn_entry_point"].split(":"); getattr(importlib.import_module(mn), fn)(u, args_cli.task, agent_cfg)
env.reset()
u.reset(env_ids=torch.arange(u.num_envs, device=u.device))
u.scene.update(u.physics_dt)   # refresh buffers only -- NO sim.step
names = robot.joint_names
jp = tt(robot.data.joint_pos); dflt = tt(robot.data.default_joint_pos)
lim = tt(robot.data.joint_pos_limits); lo, hi = lim[..., 0], lim[..., 1]
far = ((jp - dflt).abs() > 0.3).any(1)
over = (jp < lo - 0.02) | (jp > hi + 0.02)
print(f"\n[ds2] ===== {args_cli.label} =====  envs={u.num_envs}  (read BEFORE any sim step)")
print(f"[ds2] envs with a joint >0.3 rad from default (i.e. dataset state applied): {int(far.sum())} / {u.num_envs}")
print(f"[ds2] (env,joint) outside limits: {int(over.sum())} / {over.numel()}   envs with >=1: {int(over.any(1).sum())}")
print(f"[ds2] per-joint absmax BY NAME (compare to cache columns):")
for n in ["waist_yaw_joint","waist_roll_joint","waist_pitch_joint","left_ankle_roll_joint","left_ankle_pitch_joint",
          "left_elbow_joint","left_wrist_yaw_joint","right_shoulder_pitch_joint","left_knee_joint","left_hip_roll_joint"]:
    i = names.index(n)
    print(f"[ds2]   {n:26s} absmax={float(jp[:,i].abs().max()):6.3f}  limits[{float(lo[0,i]):+.3f},{float(hi[0,i]):+.3f}]"
          f"  violated={int(over[:,i].sum())}")
env.close(); simulation_app.close()
