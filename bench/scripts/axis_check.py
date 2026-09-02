#!/usr/bin/env python3
"""Render a robot in a KNOWN upright pose and report every axis convention.

If the physics says upright (projected gravity = [0,0,-1]) but the render shows
it lying down, the bug is in the viewer's reference frame, not the simulation.
"""
import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--out", type=str, default="/workspace/bench/video/axis_upright.png")
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

# Force a known-upright pose: identity quaternion, default joint positions.
st = tt(robot.data.root_state_w).clone()
st[:, 3] = 1.0            # w
st[:, 4:7] = 0.0          # x, y, z
st[:, 7:13] = 0.0         # zero velocity
robot.write_root_state_to_sim(st)
jp = tt(robot.data.default_joint_pos).clone()
robot.write_joint_state_to_sim(position=jp, velocity=torch.zeros_like(jp))
u.scene.write_data_to_sim()
u.sim.step()
u.scene.update(u.physics_dt)

pg = tt(robot.data.projected_gravity_b)[0].tolist()
quat = tt(robot.data.root_quat_w)[0].tolist()
rp = tt(robot.data.root_pos_w)[0].tolist()
print(f"[axis] PHYSICS says: root_quat_w(w,x,y,z)={[round(v,4) for v in quat]}")
print(f"[axis] PHYSICS says: projected_gravity_b={[round(v,4) for v in pg]}  (upright == [0,0,-1])")
print(f"[axis] PHYSICS says: root_pos_w={[round(v,3) for v in rp]}")

import pyglet  # noqa: E402
pyglet.options["headless"] = True
from newton.viewer import ViewerGL  # noqa: E402

from isaaclab_newton.physics import NewtonManager  # noqa: E402

model = NewtonManager.get_model()
print(f"[axis] newton model.up_axis = {getattr(model, 'up_axis', None)!r}")

viewer = ViewerGL(width=960, height=540, headless=True)
print(f"[axis] camera.up_axis BEFORE set_model = {getattr(viewer.camera, 'up_axis', None)!r}")
viewer.set_model(model)
print(f"[axis] camera.up_axis AFTER  set_model = {getattr(viewer.camera, 'up_axis', None)!r}")
viewer.set_world_offsets((0.0, 0.0, 0.0))
viewer.up_axis = 2
print(f"[axis] camera.up_axis AFTER  viewer.up_axis=2 -> {getattr(viewer.camera, 'up_axis', None)!r}")
print(f"[axis] viewer.up_axis attr = {getattr(viewer, 'up_axis', None)!r}")

# camera: level with the pelvis, looking horizontally at it
tx, ty, tz = rp
ex, ey, ez = tx + 2.5, ty - 3.0, tz + 0.3
dx, dy, dz = tx - ex, ty - ey, tz - ez
n = math.sqrt(dx*dx + dy*dy + dz*dz)
dx, dy, dz = dx/n, dy/n, dz/n
viewer.camera.fov = 55.0
viewer.set_camera(pos=wp.vec3(ex, ey, ez),
                  pitch=math.degrees(math.asin(max(-1.0, min(1.0, dz)))),
                  yaw=math.degrees(math.atan2(dy, dx)))

viewer.begin_frame(u.physics_dt)
viewer.log_state(NewtonManager.get_state())
viewer.end_frame()
frame = viewer.get_frame().numpy()
if frame.dtype != np.uint8:
    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
import subprocess
h, w = frame.shape[:2]
ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", f"{w}x{h}", "-i", "-", "-frames:v", "1", args_cli.out],
                      stdin=subprocess.PIPE)
ff.stdin.write(frame[..., :3].tobytes())
ff.stdin.close(); ff.wait()
print(f"[axis] wrote {args_cli.out} ({w}x{h})")
env.close()
simulation_app.close()
