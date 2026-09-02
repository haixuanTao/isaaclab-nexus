#!/usr/bin/env python3
"""Is the slow collapse caused by the actuators holding the robot up?

Two conditions from the same spawn state, zero policy action in both:
  pd    -- actuators as configured (PD holds the default pose)
  limp  -- actuator stiffness and damping zeroed, so joint torque ~ 0

If 'limp' collapses in a few tenths of a second and 'pd' takes seconds, the slow
descent is the actuators, and the physics is fine.
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seconds", type=float, default=3.0)
parser.add_argument("--mode", type=str, default="pd", choices=["pd", "limp"])
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import agile.isaaclab_extras.monkey_patches  # noqa: F401,E402
import agile.rl_env.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def tt(x):
    return x.torch if hasattr(x, "torch") else x


cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
cfg.seed = 42
env = gym.make(args_cli.task, cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
env.reset()

if args_cli.mode == "limp":
    n = 0
    for act in robot.actuators.values():
        for attr in ("stiffness", "damping"):
            v = getattr(act, attr, None)
            if isinstance(v, torch.Tensor):
                v.zero_()
                n += 1
        for attr in ("saturation_effort", "effort_limit"):
            v = getattr(act, attr, None)
            if isinstance(v, torch.Tensor):
                v.zero_()
    print(f"[drop] LIMP: zeroed {n} gain tensors across {len(robot.actuators)} actuator groups")

dt = u.step_dt
steps = int(args_cli.seconds / dt)
action = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
z0 = float(tt(robot.data.root_pos_w)[:, 2].mean())
print(f"[drop] mode={args_cli.mode}  z0={z0:.4f}  control dt={dt}  steps={steps}")
print(f"[drop] {'t(s)':>7} {'mean z':>9} {'drop':>8} {'freefall z':>11}")

g = abs(u.cfg.sim.gravity[2])
for i in range(steps):
    env.step(action)
    t = (i + 1) * dt
    z = float(tt(robot.data.root_pos_w)[:, 2].mean())
    if i % 5 == 0 or i == steps - 1:
        print(f"[drop] {t:7.2f} {z:9.4f} {z - z0:8.4f} {max(0.0, z0 - 0.5*g*t*t):11.4f}")

zf = float(tt(robot.data.root_pos_w)[:, 2].mean())
print(f"[drop] RESULT mode={args_cli.mode}: dropped {z0 - zf:.4f} m in {args_cli.seconds}s "
      f"(free fall would need {(2*(z0-max(zf,0.01))/g)**0.5:.3f}s for that drop)")
env.close()
simulation_app.close()
