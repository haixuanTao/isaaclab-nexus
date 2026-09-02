#!/usr/bin/env python3
"""MuJoCo combines friction with `max` unless geom priorities differ, so AGILE's
randomised foot mu (0.2-1.5) is overridden by the terrain's mu=1.0 on every
contact. Give the foot geoms a higher priority and their own mu wins.

Reports the mu actually used on foot contacts, and the fraction of foot sensor
samples reporting a pulling (negative) normal force, with and without the fix.
"""
import argparse, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--priority", action="store_true", help="apply the foot-priority fix")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app

import gymnasium as gym, numpy as np, torch, warp as wp
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_newton.physics import NewtonManager

cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.envs)
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped
sensor = u.scene.sensors["contact_forces"]
env.reset()

m = NewtonManager.get_model()
labels = list(getattr(m, "shape_label", []) or getattr(m, "shape_key", []) or [])
foot_shapes = {i for i, l in enumerate(labels) if "ankle_roll" in l.lower()}
solver = NewtonManager._solver
g2s = solver.mjc_geom_to_newton_shape.numpy()      # (worlds, geoms) -> newton shape
foot_geoms = sorted({g for w in range(g2s.shape[0]) for g in range(g2s.shape[1]) if int(g2s[w, g]) in foot_shapes})
print(f"\n[fp] foot shapes={len(foot_shapes)}  foot geoms={len(foot_geoms)}  priority_fix={args_cli.priority}", flush=True)

mjm = solver.mjw_model
if args_cli.priority and hasattr(mjm, "geom_priority"):
    pr = mjm.geom_priority.numpy()
    print(f"[fp] geom_priority before: unique={np.unique(pr)}")
    pr = pr.copy()
    if pr.ndim == 1: pr[foot_geoms] = 1
    else:            pr[:, foot_geoms] = 1
    mjm.geom_priority = wp.array(pr, dtype=mjm.geom_priority.dtype, device=mjm.geom_priority.device)
    print(f"[fp] geom_priority after:  unique={np.unique(pr)}  (feet raised)")
elif args_cli.priority:
    print("[fp] WARNING: mjw_model has no geom_priority")

sb = list(sensor.body_names)
feet_s = [i for i, n in enumerate(sb) if "ankle_roll" in n]
action = torch.zeros(u.action_space.shape, device=u.device)
mus, neg, tot, worst = [], 0, 0, 0.0
for step in range(args_cli.steps):
    env.step(action)
    mjd = solver.mjw_data
    c = NewtonManager._contacts
    cnt = int(c.rigid_contact_count.numpy()[0])
    if cnt:
        s0 = c.rigid_contact_shape0.numpy()[:cnt]; s1 = c.rigid_contact_shape1.numpy()[:cnt]
        fr = mjd.contact.friction.numpy()[:cnt]
        for k in range(cnt):
            if int(s0[k]) in foot_shapes or int(s1[k]) in foot_shapes:
                mus.append(float(fr[k][0]))
    f = sensor.data.net_forces_w
    f = f.torch if hasattr(f, "torch") else f
    fz = f[:, feet_s, 2]
    neg += int((fz < -1.0).sum()); tot += fz.numel(); worst = min(worst, float(fz.min()))

mus = np.array(mus)
if len(mus):
    print(f"[fp] mu on foot contacts: n={len(mus)} min={mus.min():.3f} median={np.median(mus):.3f} "
          f"max={mus.max():.3f} unique(first 8)={np.unique(np.round(mus,3))[:8]}")
print(f"[fp] foot sensor samples with pulling force (< -1 N): {neg}/{tot} ({100*neg/max(tot,1):.1f}%)  worst Fz={worst:.1f} N")
env.close(); simulation_app.close()
