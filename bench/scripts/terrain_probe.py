#!/usr/bin/env python3
"""Sample the terrain mesh and report whether tiles are flat plates or domes."""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--num_envs", type=int, default=4)
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

cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
env = gym.make(args_cli.task, cfg=cfg)
u = env.unwrapped
terrain = u.scene.terrain

verts = None
for attr in ("meshes", "terrain_meshes", "warp_meshes"):
    obj = getattr(terrain, attr, None)
    if obj is None:
        continue
    cand = list(obj.values()) if isinstance(obj, dict) else list(obj)
    for m in cand:
        for vattr in ("vertices", "points", "v"):
            v = getattr(m, vattr, None)
            if v is not None:
                verts = np.asarray(v)
                break
        if verts is not None:
            break
    if verts is not None:
        print(f"[terr] got vertices from terrain.{attr}")
        break

if verts is None:
    print("[terr] could not reach terrain vertices; reporting sensor heights instead")
else:
    print(f"[terr] terrain vertices: {verts.shape}")
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    print(f"[terr] x range [{x.min():.2f}, {x.max():.2f}]  y range [{y.min():.2f}, {y.max():.2f}]")
    print(f"[terr] z range [{z.min():.3f}, {z.max():.3f}]  mean {z.mean():.3f}  std {z.std():.3f}")
    uz, cnt = np.unique(np.round(z, 3), return_counts=True)
    order = np.argsort(-cnt)[:12]
    print(f"[terr] distinct rounded heights: {len(uz)}")
    print("[terr] most common heights (height: vertex count):")
    for i in order:
        print(f"[terr]    {uz[i]:8.3f} : {cnt[i]:8d}")
    frac = cnt[order].sum() / len(z)
    print(f"[terr] top-12 heights cover {100*frac:.1f}% of all vertices")
    print(f"[terr] => {'PIECEWISE-FLAT plates (few discrete heights)' if frac > 0.5 else 'SMOOTHLY VARYING surface (domes/bumps)'}")

env.close()
simulation_app.close()
