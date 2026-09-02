#!/usr/bin/env python3
"""Record the Newton-trained G1 policy by driving Newton's ViewerGL directly.

Isaac Lab's video recorder only tracks the robot when a Newton visualizer is
live, and it did not sync in this headless setup -- every frame came out as
empty terrain. This bypasses that path: it owns the viewer, and pins the camera
to the env-0 robot's actual root position every frame.

Frames are piped straight to ffmpeg.
"""

import argparse
import os
import subprocess
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--num_envs", type=int, default=9)
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--keep-assist", action="store_true", help="keep the training-only lift/harness actions (default: removed, as in eval.py)")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--cam_offset", type=str, default="2.2,-3.0,1.2")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

os.environ.setdefault("PYGLET_HEADLESS", "1")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import agile.isaaclab_extras.monkey_patches  # noqa: F401,E402
import agile.rl_env.tasks  # noqa: F401,E402
from agile.rl_env.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    if not args_cli.keep_assist:
        from agile.rl_env.rsl_rl.export_pruning import prepare_training_only_actions_for_evaluation
        _removed, _held = prepare_training_only_actions_for_evaluation(env_cfg)
        print(f"[assist] removed training-only actions {_removed}; held at default {_held}", flush=True)

    env_cfg.seed = 42
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    # The raw env hands back a TensorDict the TorchScript policy won't accept;
    # this is the same wrapper eval.py uses to get a flat observation tensor.
    env = RslRlVecEnvWrapper(env)

    policy = torch.jit.load(args_cli.checkpoint, map_location=unwrapped.device).eval()

    import pyglet

    pyglet.options["headless"] = True
    from newton.viewer import ViewerGL

    from isaaclab_newton.physics import NewtonManager

    viewer = ViewerGL(width=args_cli.width, height=args_cli.height, headless=True)
    viewer.set_model(NewtonManager.get_model())
    viewer.set_world_offsets((0.0, 0.0, 0.0))
    viewer.up_axis = 2
    viewer.camera.fov = 60.0

    ox, oy, oz = (float(v) for v in args_cli.cam_offset.split(","))
    cam_z_ref = {"z": None}  # fixed eye height, set from the robot's initial height

    def aim_at(target):
        tx, ty, tz = float(target[0]), float(target[1]), float(target[2])
        # Track horizontally only. The eye height is pinned to the robot's initial
        # height so vertical motion is visible against a static horizon instead of
        # being cancelled out by a camera that falls with the robot.
        if cam_z_ref["z"] is None:
            cam_z_ref["z"] = tz
        ex, ey, ez = tx + ox, ty + oy, cam_z_ref["z"] + oz
        dx, dy, dz = tx - ex, ty - ey, tz - ez
        n = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
        dx, dy, dz = dx / n, dy / n, dz / n
        import warp as wp

        viewer.set_camera(
            pos=wp.vec3(ex, ey, ez),
            pitch=math.degrees(math.asin(max(-1.0, min(1.0, dz)))),
            yaw=math.degrees(math.atan2(dy, dx)),
        )

    dt = unwrapped.step_dt
    fps = int(round(1.0 / dt))
    n_steps = int(round(args_cli.seconds / dt))

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{args_cli.width}x{args_cli.height}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", args_cli.out],
        stdin=subprocess.PIPE,
    )

    obs, _ = env.reset()
    print(f"[rec] obs type={type(obs).__name__} {n_steps} steps @ {fps} fps -> {args_cli.out}", flush=True)
    written = 0
    for step in range(n_steps):
        with torch.inference_mode():
            po = obs["policy"] if (hasattr(obs, "keys") and "policy" in obs.keys()) else obs
            po = po.torch if hasattr(po, "torch") else po
            action = policy(torch.as_tensor(po))
        obs, _, _, _ = env.step(action)

        root = robot.data.root_pos_w
        root = root.torch if hasattr(root, "torch") else root
        aim_at(root[0].tolist())

        viewer.begin_frame(dt)
        viewer.log_state(NewtonManager.get_state())
        viewer.end_frame()
        frame = viewer.get_frame().numpy()
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
        ff.stdin.write(frame[..., :3].tobytes())
        written += 1
        if step % 100 == 0:
            print(f"[rec] step {step}/{n_steps}  robot0 z={float(root[0][2]):.3f}", flush=True)

    ff.stdin.close()
    ff.wait()
    print(f"[rec] wrote {written} frames -> {args_cli.out}", flush=True)
    env.close()
    simulation_app.close()


main()
