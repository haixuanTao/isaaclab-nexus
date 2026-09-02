#!/usr/bin/env python3
"""Sign convention of projected_gravity_b, with the root forced to identity.

Upright + identity quaternion => gravity in body frame must be [0, 0, -1].
Runs on either engine; PhysX is the reference.
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--num_envs", type=int, default=4)
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

print(f"\n[pg] ===== {args_cli.label} =====")
print(f"[pg] sim gravity = {u.cfg.sim.gravity}")

for name, q in (("identity (upright)", [1.0, 0.0, 0.0, 0.0]),
                ("180deg roll (upside down)", [0.0, 1.0, 0.0, 0.0])):
    st = tt(robot.data.root_state_w).clone()
    st[:, 2] += 3.0
    st[:, 3] = q[0]; st[:, 4] = q[1]; st[:, 5] = q[2]; st[:, 6] = q[3]
    st[:, 7:13] = 0.0
    robot.write_root_state_to_sim(st)
    jp = tt(robot.data.default_joint_pos).clone()
    robot.write_joint_state_to_sim(position=jp, velocity=torch.zeros_like(jp))
    u.scene.write_data_to_sim()
    u.sim.step()
    u.scene.update(u.physics_dt)
    pg = tt(robot.data.projected_gravity_b)[0].tolist()
    qq = tt(robot.data.root_quat_w)[0].tolist()
    print(f"[pg] {name:28s} quat(w,x,y,z)={[round(v,3) for v in qq]}  "
          f"projected_gravity_b={[round(v,4) for v in pg]}")

env.close()
simulation_app.close()
