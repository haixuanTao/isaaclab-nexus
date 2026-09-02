#!/usr/bin/env python3
"""Decisive free-fall test: is the robot ACTUALLY airborne while we measure it?

The earlier probe reported ~-1 m/s^2 on both engines. Either both are wrong, or
the teleported robot was never in free fall. This reports contact forces and the
lowest body height alongside z(t), and compares against the analytic 0.5*g*t^2,
so the two explanations can be told apart.
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--lift", type=float, default=20.0)
parser.add_argument("--steps", type=int, default=60)
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
env.reset()
dt = u.physics_dt
g = abs(u.cfg.sim.gravity[2])

# Lift high enough that contact is impossible, and zero all velocities.
st = tt(robot.data.root_state_w).clone()
st[:, 2] += args_cli.lift
st[:, 7:13] = 0.0
robot.write_root_state_to_sim(st)
jp = tt(robot.data.joint_pos).clone()
jv = torch.zeros_like(tt(robot.data.joint_vel))
robot.write_joint_state_to_sim(position=jp, velocity=jv)
u.scene.write_data_to_sim()
u.sim.step()
u.scene.update(dt)

z0 = float(tt(robot.data.root_pos_w)[:, 2].mean())
print(f"\n[ff] ===== {args_cli.label} =====")
print(f"[ff] lift={args_cli.lift} m   z0 after teleport = {z0:.4f}   dt={dt}  g={g}")
print(f"[ff] {'t(s)':>7} {'z':>9} {'vz':>9} {'z_expect':>9} {'err':>8} {'minBodyZ':>9} {'contactN':>10}")

sensor = u.scene.sensors.get("contact_forces")
for i in range(args_cli.steps):
    u.sim.step()
    u.scene.update(dt)
    t = (i + 1) * dt
    z = float(tt(robot.data.root_pos_w)[:, 2].mean())
    vz = float(tt(robot.data.root_lin_vel_w)[:, 2].mean())
    bz = float(tt(robot.data.body_pos_w)[..., 2].min())
    cf = 0.0
    if sensor is not None:
        f = tt(sensor.data.net_forces_w)
        if f is not None:
            cf = float(f.norm(dim=-1).max())
    z_exp = z0 - 0.5 * g * t * t
    if i % 6 == 0 or i == args_cli.steps - 1:
        print(f"[ff] {t:7.3f} {z:9.4f} {vz:9.4f} {z_exp:9.4f} {z - z_exp:8.4f} {bz:9.3f} {cf:10.2f}")

t_end = args_cli.steps * dt
z_end = float(tt(robot.data.root_pos_w)[:, 2].mean())
vz_end = float(tt(robot.data.root_lin_vel_w)[:, 2].mean())
a_eff = 2.0 * (z_end - z0) / (t_end * t_end)
print(f"\n[ff] over {t_end:.3f}s: dz={z_end - z0:+.4f} m   vz_end={vz_end:+.4f} m/s")
print(f"[ff] effective accel from displacement = {a_eff:8.3f} m/s^2   (expected {-g:.3f})")
print(f"[ff] effective accel from final vel    = {vz_end / t_end:8.3f} m/s^2")
print(f"[ff] VERDICT: {'FREE FALL OK' if abs(a_eff + g) < 2.0 else 'NOT FREE-FALLING (check contactN / minBodyZ above)'}")
env.close()
simulation_app.close()
