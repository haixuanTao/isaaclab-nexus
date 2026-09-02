#!/usr/bin/env python3
"""Is weight consistent with contact, and can a torque-free robot hold a tilt?

Test A: standing -- total vertical contact force vs m*g. These must match.
Test B: tilt the robot 40 deg with all actuator torque removed and watch whether
        it topples. A torque-free body at 40 deg MUST fall over.
"""
import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--tilt_deg", type=float, default=40.0)
parser.add_argument("--label", type=str, default="engine")
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
sensor = u.scene.sensors.get("contact_forces")
g = abs(u.cfg.sim.gravity[2])
env.reset()

action = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
for _ in range(25):
    env.step(action)

mass = None
for attr in ("default_mass", "body_mass"):
    m = tt(getattr(robot.data, attr, None))
    if m is not None and m.numel():
        mass = m
        print(f"[bal] masses from robot.data.{attr} shape={tuple(m.shape)}")
        break

print(f"\n[bal] ===== {args_cli.label} =====")
if mass is not None:
    total_m = float(mass[0].sum()) if mass.dim() > 1 else float(mass.sum())
    print(f"[bal] total robot mass = {total_m:.2f} kg   weight = {total_m*g:.1f} N")
else:
    total_m = float("nan")
    print("[bal] mass unavailable")

if sensor is not None:
    f = tt(sensor.data.net_forces_w)
    fz = f[..., 2].sum(dim=-1)
    print(f"[bal] total vertical contact force: mean {float(fz.mean()):8.1f} N   "
          f"median {float(fz.median()):8.1f} N")
    if mass is not None and total_m == total_m:
        print(f"[bal] ratio contact/weight = {float(fz.mean())/(total_m*g):.3f}  (1.0 == consistent)")

# ---- Test B: torque-free tilt ----
n = 0
for act in robot.actuators.values():
    for attr in ("stiffness", "damping", "saturation_effort", "effort_limit"):
        v = getattr(act, attr, None)
        if isinstance(v, torch.Tensor):
            v.zero_()
            n += 1
print(f"\n[bal] zeroed {n} actuator tensors -> torque-free")

th = math.radians(args_cli.tilt_deg)
st = tt(robot.data.root_state_w).clone()
st[:, 3] = math.cos(th / 2)   # w
st[:, 4] = 0.0
st[:, 5] = math.sin(th / 2)   # pitch about y
st[:, 6] = 0.0
st[:, 2] += 0.25
st[:, 7:13] = 0.0
robot.write_root_state_to_sim(st)
u.scene.write_data_to_sim()
u.sim.step()
u.scene.update(u.physics_dt)

print(f"[bal] tilted {args_cli.tilt_deg} deg, torque-free. Tracking uprightness:")
print(f"[bal] {'t(s)':>6} {'projG_z':>9} {'tilt_deg':>9} {'z':>8}")
dt = u.step_dt
for i in range(100):
    env.step(action)
    if i % 12 == 0 or i == 99:
        pg = tt(robot.data.projected_gravity_b)[:, 2].mean()
        ang = math.degrees(math.acos(max(-1.0, min(1.0, -float(pg)))))
        z = float(tt(robot.data.root_pos_w)[:, 2].mean())
        print(f"[bal] {(i+1)*dt:6.2f} {float(pg):9.4f} {ang:9.2f} {z:8.4f}")

pg_end = float(tt(robot.data.projected_gravity_b)[:, 2].mean())
ang_end = math.degrees(math.acos(max(-1.0, min(1.0, -pg_end))))
print(f"\n[bal] final tilt = {ang_end:.1f} deg (started {args_cli.tilt_deg})")
print(f"[bal] VERDICT: {'TOPPLED as expected' if ang_end > args_cli.tilt_deg + 15 else 'DID NOT TOPPLE -- torque-free body holding a tilt is unphysical'}")
env.close()
simulation_app.close()
