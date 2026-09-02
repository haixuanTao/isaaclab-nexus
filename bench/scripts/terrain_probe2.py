#!/usr/bin/env python3
"""Sample terrain height via raycast and decide: flat plates or smooth domes?"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402

import agile.isaaclab_extras.monkey_patches  # noqa: F401,E402
import agile.rl_env.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def tt(x):
    return x.torch if hasattr(x, "torch") else x


cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
env = gym.make(args_cli.task, cfg=cfg)
u = env.unwrapped
env.reset()
for _ in range(3):
    u.sim.step()
    u.scene.update(u.physics_dt)

print("[terr] scene sensors:", list(u.scene.sensors.keys()))
sensor = None
for name in u.scene.sensors:
    if "height" in name or "ray" in name:
        sensor = u.scene.sensors[name]
        print(f"[terr] using sensor '{name}'")
        break

if sensor is None:
    print("[terr] no raycast sensor available")
else:
    hits = tt(sensor.data.ray_hits_w)
    h = hits[..., 2].detach().cpu().numpy().ravel()
    h = h[np.isfinite(h)]
    print(f"[terr] {h.size} terrain height samples")
    print(f"[terr] z range [{h.min():.3f}, {h.max():.3f}]  mean {h.mean():.3f}  std {h.std():.3f}")
    # quantise: piecewise-flat terrain concentrates on few discrete heights
    uz, cnt = np.unique(np.round(h, 2), return_counts=True)
    order = np.argsort(-cnt)
    top = order[:10]
    print(f"[terr] distinct heights (2dp): {len(uz)}")
    print("[terr] most common heights (m : share of samples):")
    for i in top:
        print(f"[terr]    {uz[i]:8.2f} : {100*cnt[i]/h.size:6.2f}%")
    share = cnt[top].sum() / h.size
    print(f"[terr] top-10 discrete heights cover {100*share:.1f}% of samples")
    print(f"[terr] VERDICT: {'PIECEWISE-FLAT plates -> the domed look is viewer SHADING' if share > 0.5 else 'SMOOTH/BUMPY surface -> geometry really is curved'}")

env.close()
simulation_app.close()
