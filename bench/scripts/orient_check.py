#!/usr/bin/env python3
"""Which way is a written identity quaternion actually facing?

Writes identity, lifts 3 m, then STEPS (defeating the projected_gravity cache)
and watches where it lands. Upright lands on its feet with pelvis ~0.8 m and
projected_gravity_b -> [0,0,-1]. Inverted lands on its head.
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--num_envs", type=int, default=8)
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
print(f"\n[or] ===== {args_cli.label} =====")

st = tt(robot.data.root_state_w).clone()
st[:, 2] += 3.0
st[:, 3] = 1.0; st[:, 4:7] = 0.0     # identity quaternion
st[:, 7:13] = 0.0
robot.write_root_state_to_sim(st)
jp = tt(robot.data.default_joint_pos).clone()
robot.write_joint_state_to_sim(position=jp, velocity=torch.zeros_like(jp))
u.scene.write_data_to_sim()

z_origin = tt(u.scene.env_origins)[:, 2]
print(f"[or] wrote identity quat, lifted 3 m. Stepping (cache now refreshes):")
print(f"[or] {'t(s)':>6} {'quat_w':>8} {'pg_z':>8} {'pelvis_z':>9} {'head_z':>8} {'foot_z':>8}")

names = robot.body_names
head_i = next((i for i, n in enumerate(names) if "head" in n.lower() or "torso" in n.lower()), 0)
foot_i = next((i for i, n in enumerate(names) if "ankle_roll" in n.lower()), len(names) - 1)

dt = u.step_dt
action = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
for i in range(90):
    env.step(action)
    if i % 12 == 0 or i == 89:
        q = tt(robot.data.root_quat_w)[:, 0].mean()
        pg = tt(robot.data.projected_gravity_b)[:, 2].mean()
        pz = (tt(robot.data.root_pos_w)[:, 2] - z_origin).mean()
        bp = tt(robot.data.body_pos_w)[..., 2]
        hz = (bp[:, head_i] - z_origin).mean()
        fz = (bp[:, foot_i] - z_origin).mean()
        print(f"[or] {(i+1)*dt:6.2f} {float(q):8.3f} {float(pg):8.3f} {float(pz):9.3f} "
              f"{float(hz):8.3f} {float(fz):8.3f}")

bp = tt(robot.data.body_pos_w)[..., 2]
hz = float((bp[:, head_i] - z_origin).mean())
fz = float((bp[:, foot_i] - z_origin).mean())
print(f"[or] final: '{names[head_i]}' z={hz:.3f}   '{names[foot_i]}' z={fz:.3f}")
print(f"[or] VERDICT: {'UPRIGHT (torso above feet)' if hz > fz else 'INVERTED (feet above torso)'}")
env.close()
simulation_app.close()
