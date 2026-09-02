#!/usr/bin/env python3
"""Measure gravity empirically under Newton: drop the robots and fit z(t).

Lifts every robot 5 m, zeroes velocity, steps with zero action, and compares the
observed vertical acceleration against the configured -9.81 m/s^2.
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--num_envs", type=int, default=16)
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

print(f"[grav] cfg.sim.gravity        = {u.cfg.sim.gravity}")
try:
    from isaaclab_newton.physics import NewtonManager
    m = NewtonManager.get_model()
    print(f"[grav] newton model.gravity   = {getattr(m, 'gravity', None)}")
except Exception as e:
    print(f"[grav] (newton model gravity unavailable: {e})")

env.reset()
dt = u.physics_dt
print(f"[grav] physics dt = {dt}   control dt = {u.step_dt}")

# lift all robots 5 m and zero their velocity, then let them fall untouched
state = tt(robot.data.root_state_w).clone()
state[:, 2] += 5.0
state[:, 7:13] = 0.0
robot.write_root_state_to_sim(state)
u.sim.step()
u.scene.update(dt)

zs, vs = [], []
action = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
for i in range(30):
    u.action_manager.process_action(action)
    u.action_manager.apply_action()
    u.scene.write_data_to_sim()
    u.sim.step()
    u.scene.update(dt)
    zs.append(float(tt(robot.data.root_pos_w)[:, 2].mean()))
    vs.append(float(tt(robot.data.root_lin_vel_w)[:, 2].mean()))

print(f"\n[grav] {'step':>5} {'mean z':>10} {'mean vz':>10} {'dvz/dt':>10}")
for i in range(0, len(zs), 3):
    a = (vs[i] - vs[i - 1]) / dt if i > 0 else float("nan")
    print(f"[grav] {i:5d} {zs[i]:10.4f} {vs[i]:10.4f} {a:10.3f}")

acc = [(vs[i] - vs[i - 1]) / dt for i in range(2, len(vs))]
acc_med = sorted(acc)[len(acc) // 2]
print(f"\n[grav] measured vertical acceleration (median) = {acc_med:.3f} m/s^2")
print(f"[grav] configured gravity z                    = {u.cfg.sim.gravity[2]:.3f} m/s^2")
print(f"[grav] VERDICT: {'gravity CORRECT (falls downward at ~9.81)' if -12 < acc_med < -7 else 'GRAVITY WRONG'}")
env.close()
simulation_app.close()
