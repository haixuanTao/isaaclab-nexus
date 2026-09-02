#!/usr/bin/env python3
"""Newton-side shape materials actually in the model: friction mu, restitution,
contact ke/kd, for the feet shapes and the terrain shape. The task's terrain
authors restitution=1.0 with 'multiply' combine; if Newton applies 1.0 the
contacts are perfectly elastic."""
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
def g(n):
    a = getattr(m, n, None)
    try: return wp.to_torch(a).float().cpu().numpy() if isinstance(a, wp.array) else (np.asarray(a) if a is not None else None)
    except Exception: return None
keys = [k for k in ("shape_material_mu","shape_material_restitution","shape_material_ke","shape_material_kd","shape_material_kf","shape_material_ka") if g(k) is not None]
labels = list(getattr(m, "shape_label", []) or getattr(m, "shape_key", []) or [])
print(f"\n[sm] shapes={m.shape_count}  fields={keys}  labels={len(labels)}")
def show(idx, tag):
    for i in idx[:2]:
        print(f"[sm] {tag:8s} {(labels[i] if i < len(labels) else str(i))[-46:]:46s} " + "  ".join(f"{k.split('_')[-1]}={float(g(k)[i]):.4g}" for k in keys))
feet = [i for i, l in enumerate(labels) if "ankle_roll" in l]; terr = [i for i, l in enumerate(labels) if "terrain" in l.lower() or "ground" in l.lower()]
show(feet, "FOOT"); show(terr, "TERRAIN")
for k in keys:
    a = g(k); print(f"[sm] {k:28s} min={a.min():.4g} max={a.max():.4g} unique={np.unique(np.round(a,3))[:6]}")
env.close(); simulation_app.close()
