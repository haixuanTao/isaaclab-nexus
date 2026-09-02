#!/usr/bin/env python3
"""Log commanded pelvis height vs actual during a trained-policy rollout.

If the 'fall' in the video is really the policy tracking a descending height
command, commanded and actual will move together.
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--num_envs", type=int, default=9)
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--checkpoint", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import agile.isaaclab_extras.monkey_patches  # noqa: F401,E402
import agile.rl_env.tasks  # noqa: F401,E402
from agile.rl_env.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def tt(x):
    return x.torch if hasattr(x, "torch") else x


cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
cfg.seed = 42
raw = gym.make(args_cli.task, cfg=cfg)
u = raw.unwrapped
robot = u.scene["robot"]
env = RslRlVecEnvWrapper(raw)
policy = torch.jit.load(args_cli.checkpoint, map_location=u.device).eval()

print("[hc] command terms:", list(u.command_manager._terms.keys()))
obs, _ = env.reset()
dt = u.step_dt
steps = int(args_cli.seconds / dt)
print(f"[hc] {'t(s)':>6} {'cmd_h':>9} {'actual_z':>9} {'err':>8}")
for i in range(steps):
    with torch.inference_mode():
        po = obs["policy"] if (hasattr(obs, "keys") and "policy" in obs.keys()) else obs
        po = po.torch if hasattr(po, "torch") else po
        a = policy(torch.as_tensor(po))
    obs, *_ = env.step(a)
    if i % 25 == 0 or i == steps - 1:
        cmd = None
        for name in u.command_manager._terms:
            c = tt(u.command_manager.get_command(name))
            if c is not None and c.numel():
                cmd = float(c[0].flatten()[0])
                break
        z = float(tt(robot.data.root_pos_w)[0, 2])
        print(f"[hc] {(i+1)*dt:6.2f} {cmd if cmd is not None else float('nan'):9.4f} {z:9.4f} "
              f"{(z - cmd) if cmd is not None else float('nan'):8.4f}")
raw.close()
simulation_app.close()
