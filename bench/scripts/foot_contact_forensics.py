#!/usr/bin/env python3
"""Where do the negative foot 'normal forces' come from?

For every rigid contact touching a foot shape, record the force AS APPLIED TO THE
FOOT (+force if the foot is shape0, -force if shape1 -- the same rule Newton's
accumulate_contact_forces_kernel uses) together with the contact normal and the
counterpart shape. Then ask: do the negative-Fz contacts have a downward normal,
and what are they against -- terrain top, terrain underside, or another robot part?
"""
import argparse, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app

import gymnasium as gym, numpy as np, torch
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_newton.physics import NewtonManager

cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.envs)
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped
env.reset()

m = NewtonManager.get_model()
labels = list(getattr(m, "shape_label", []) or getattr(m, "shape_key", []) or [])
foot_shapes = {i for i, l in enumerate(labels) if "ankle_roll" in l.lower()}
print(f"\n[fcf] shapes={len(labels)}  foot shapes={len(foot_shapes)}", flush=True)

def lab(i):
    if i < 0 or i >= len(labels): return f"shape{i}"
    s = labels[i]
    return s[-46:]

action = torch.zeros(u.action_space.shape, device=u.device)
neg_rows, pos_rows = [], []
n_foot_contacts = 0

for step in range(args_cli.steps):
    env.step(action)
    c = NewtonManager._contacts
    if c is None or c.force is None:
        print("[fcf] no contacts/force array -- report_contacts off"); break
    cnt = int(c.rigid_contact_count.numpy()[0])
    if cnt == 0: continue
    s0 = c.rigid_contact_shape0.numpy()[:cnt]
    s1 = c.rigid_contact_shape1.numpy()[:cnt]
    fw = c.force.numpy()[:cnt]        # spatial: (force, torque)
    nw = c.rigid_contact_normal.numpy()[:cnt]
    f_lin = fw[:, :3]                 # spatial_top == linear force [N]

    for k in range(cnt):
        a, b = int(s0[k]), int(s1[k])
        a_foot, b_foot = a in foot_shapes, b in foot_shapes
        if not (a_foot or b_foot): continue
        n_foot_contacts += 1
        # force as applied to the foot, matching the sensor's sign rule
        sgn = 1.0 if a_foot else -1.0
        Ff = sgn * f_lin[k]
        nz = float(nw[k][2]); fz = float(Ff[2])
        other = b if a_foot else a
        row = (fz, nz, sgn * nw[k][2], lab(other), step)
        (neg_rows if fz < -1.0 else pos_rows).append(row)

print(f"[fcf] foot contacts sampled: {n_foot_contacts}   with Fz < -1 N: {len(neg_rows)} "
      f"({100*len(neg_rows)/max(n_foot_contacts,1):.1f}%)", flush=True)

def summarize(rows, name):
    if not rows:
        print(f"[fcf] {name}: none"); return
    fz = np.array([r[0] for r in rows]); nz = np.array([r[1] for r in rows])
    snz = np.array([r[2] for r in rows])
    print(f"[fcf] {name}: n={len(rows)}  Fz min={fz.min():.1f} max={fz.max():.1f} mean={fz.mean():.1f}")
    print(f"[fcf]   raw normal.z:            min={nz.min():.3f} median={np.median(nz):.3f} max={nz.max():.3f}"
          f"   fraction nz<0: {100*(nz<0).mean():.1f}%")
    print(f"[fcf]   normal.z oriented->foot: min={snz.min():.3f} median={np.median(snz):.3f} max={snz.max():.3f}"
          f"   fraction <0: {100*(snz<0).mean():.1f}%")
    from collections import Counter
    for lbl, n in Counter(r[3] for r in rows).most_common(6):
        print(f"[fcf]   counterpart {n:6d}x  {lbl}")

summarize(neg_rows, "NEGATIVE-Fz foot contacts")
summarize(pos_rows, "positive-Fz foot contacts")

if neg_rows:
    print("[fcf] worst 10 negative contacts (Fz, raw nz, oriented nz, counterpart, step):")
    for r in sorted(neg_rows)[:10]:
        print(f"[fcf]   Fz={r[0]:10.1f}  nz={r[1]:+.3f}  nz->foot={r[2]:+.3f}  {r[3]}  step={r[4]}")

env.close(); simulation_app.close()
