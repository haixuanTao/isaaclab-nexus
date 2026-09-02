#!/usr/bin/env python3
"""Terrain mesh as Newton built it: fraction of faces whose normal points DOWN
(z<0) and winding consistency. A pulling contact normal is the fingerprint of an
inverted face."""
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
AppLauncher.add_app_launcher_args(parser); args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, numpy as np, warp as wp
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_newton.physics import NewtonManager
cfg = parse_env_cfg(args_cli.task, num_envs=2); env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; env.reset()
m = NewtonManager.get_model()
labels = list(getattr(m, "shape_label", []) or getattr(m, "shape_key", []) or [])
ti = [i for i, l in enumerate(labels) if "terrain" in l.lower()]
print(f"\n[nrm] terrain shapes: {[(i, labels[i][-30:]) for i in ti][:3]}")
src = m.shape_source[ti[0]]
V = np.asarray(src.vertices if not isinstance(src.vertices, wp.array) else src.vertices.numpy(), dtype=np.float64)
I = np.asarray(src.indices if not isinstance(src.indices, wp.array) else src.indices.numpy(), dtype=np.int64).reshape(-1, 3)
print(f"[nrm] vertices={len(V)} faces={len(I)}  z range [{V[:,2].min():.3f}, {V[:,2].max():.3f}]")
a, b, c = V[I[:,0]], V[I[:,1]], V[I[:,2]]
n = np.cross(b - a, c - a); area = np.linalg.norm(n, axis=1); ok = area > 1e-12; n = n[ok] / area[ok, None]
down = n[:, 2] < -1e-6; flat_down = n[:, 2] < -0.5
print(f"[nrm] faces with normal.z < 0 (pointing DOWN): {down.sum()} / {len(n)} = {100*down.mean():.2f}%   strongly down (nz<-0.5): {flat_down.sum()} ({100*flat_down.mean():.2f}%)")
print(f"[nrm] normal.z distribution: min={n[:,2].min():.3f}  p1={np.percentile(n[:,2],1):.3f}  median={np.median(n[:,2]):.3f}  max={n[:,2].max():.3f}")
print(f"[nrm] degenerate faces: {(~ok).sum()}")
# scale / transform sanity
try:
    tf = m.shape_transform.numpy()[ti[0]]; sc = m.shape_scale.numpy()[ti[0]]
    print(f"[nrm] shape_transform pos={np.round(tf[:3],3)} quat={np.round(tf[3:],3)}  scale={np.round(sc,3)}")
except Exception as e: print("[nrm] transform read failed:", e)
env.close(); simulation_app.close()
