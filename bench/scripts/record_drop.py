#!/usr/bin/env python3
"""Record the free-fall drop that showed Newton reorienting mid-air.

Identity quaternion + default joint pose, lifted, then released with zero action.
Camera is fixed (no tracking) so rotation and descent are both plainly visible.
"""
import argparse
import math
import os
import subprocess
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--lift", type=float, default=3.0)
parser.add_argument("--seconds", type=float, default=3.0)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
os.environ.setdefault("PYGLET_HEADLESS", "1")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

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

st = tt(robot.data.root_state_w).clone()
st[:, 2] += args_cli.lift
st[:, 3] = 1.0; st[:, 4:7] = 0.0
st[:, 7:13] = 0.0
robot.write_root_state_to_sim(st)
jp = tt(robot.data.default_joint_pos).clone()
robot.write_joint_state_to_sim(position=jp, velocity=torch.zeros_like(jp))
u.scene.write_data_to_sim()

import pyglet  # noqa: E402
pyglet.options["headless"] = True
from newton.viewer import ViewerGL  # noqa: E402

from isaaclab_newton.physics import NewtonManager  # noqa: E402

viewer = ViewerGL(width=args_cli.width, height=args_cli.height, headless=True)
viewer.set_model(NewtonManager.get_model())
viewer.set_world_offsets((0.0, 0.0, 0.0))
viewer.up_axis = 2
viewer.camera.fov = 50.0

# Fixed camera aimed at the middle of the fall.
rp = tt(robot.data.root_pos_w)[0].tolist()
tx, ty, tz = rp[0], rp[1], rp[2] - args_cli.lift * 0.5
ex, ey, ez = tx + 4.0, ty - 5.0, tz + 0.8
dx, dy, dz = tx - ex, ty - ey, tz - ez
n = math.sqrt(dx * dx + dy * dy + dz * dz)
dx, dy, dz = dx / n, dy / n, dz / n
viewer.set_camera(pos=wp.vec3(ex, ey, ez),
                  pitch=math.degrees(math.asin(max(-1.0, min(1.0, dz)))),
                  yaw=math.degrees(math.atan2(dy, dx)))

dt = u.step_dt
fps = int(round(1.0 / dt))
n_steps = int(round(args_cli.seconds / dt))
ff = subprocess.Popen(
    ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{args_cli.width}x{args_cli.height}", "-r", str(fps), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", args_cli.out],
    stdin=subprocess.PIPE)

action = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
print(f"[drop] {n_steps} frames @ {fps} fps -> {args_cli.out}", flush=True)
for i in range(n_steps):
    env.step(action)
    viewer.begin_frame(dt)
    viewer.log_state(NewtonManager.get_state())
    viewer.end_frame()
    fr = viewer.get_frame().numpy()
    if fr.dtype != np.uint8:
        fr = (np.clip(fr, 0, 1) * 255).astype(np.uint8)
    ff.stdin.write(fr[..., :3].tobytes())
    if i % 25 == 0:
        q = float(tt(robot.data.root_quat_w)[0, 0])
        z = float(tt(robot.data.root_pos_w)[0, 2])
        print(f"[drop] t={i*dt:5.2f}  quat_w={q:+.3f}  z={z:6.3f}", flush=True)

ff.stdin.close(); ff.wait()
print(f"[drop] wrote {args_cli.out}", flush=True)
env.close()
simulation_app.close()
