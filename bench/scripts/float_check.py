#!/usr/bin/env python3
"""How far above the terrain does the lowest link sit at rest?

Terrain here is flat plates 0.000-0.016 m. If a settled robot's lowest body sits
well above that, its collision geometry is inflated relative to the visual mesh.
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--num_envs", type=int, default=32)
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
action = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
for _ in range(50):          # let it settle under default PD
    env.step(action)

bp = tt(robot.data.body_pos_w)                      # (E, B, 3)
env_origin_z = tt(u.scene.env_origins)[:, 2]        # terrain offset per env
low = bp[..., 2].min(dim=1).values - env_origin_z   # lowest link above its env origin
names = robot.body_names
idx = bp[..., 2].min(dim=1).indices

print(f"\n[float] ===== {args_cli.label} =====")
print(f"[float] lowest-link height above env origin: mean {float(low.mean()):.4f} m  "
      f"median {float(low.median()):.4f}  min {float(low.min()):.4f}  max {float(low.max()):.4f}")
from collections import Counter
c = Counter(names[int(i)] for i in idx)
print(f"[float] lowest body is usually: {c.most_common(3)}")
print(f"[float] pelvis (root) z above origin: {float((tt(robot.data.root_pos_w)[:,2]-env_origin_z).mean()):.4f} m")
print("[float] (terrain relief here is 0.000-0.016 m, so a settled foot should be near 0)")
env.close()
simulation_app.close()
