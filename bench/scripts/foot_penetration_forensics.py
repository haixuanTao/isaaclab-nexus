#!/usr/bin/env python3
"""Are the negative foot forces caused by the foot penetrating the terrain?

Newton's contact index == MuJoCo-Warp's contact index (the conversion kernel is
launched over naconmax and writes slot-for-slot), so per foot contact we can pair
Newton's shape ids and force with MJWarp's own contact.dist (penetration depth,
negative = interpenetrating) and frame. Then split by force sign.
"""
import argparse, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--flat", action="store_true")
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
if args_cli.flat:
    # flat plane: no triangle edges for the foot spheres to catch on
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    print("[pen] FLAT PLANE terrain", flush=True)
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped
robot = u.scene["robot"]
env.reset()

m = NewtonManager.get_model()
labels = list(getattr(m, "shape_label", []) or getattr(m, "shape_key", []) or [])
foot_shapes = {i for i, l in enumerate(labels) if "ankle_roll" in l.lower()}
solver = NewtonManager._solver
print(f"\n[pen] foot shapes={len(foot_shapes)}  solver={type(solver).__name__}", flush=True)

action = torch.zeros(u.action_space.shape, device=u.device)
rows = []
for step in range(args_cli.steps):
    env.step(action)
    c = NewtonManager._contacts
    mjd = getattr(solver, "mjw_data", None)
    if c is None or c.force is None or mjd is None:
        print("[pen] missing contacts or mjw_data"); break
    cnt = int(c.rigid_contact_count.numpy()[0])
    if cnt == 0: continue
    s0 = c.rigid_contact_shape0.numpy()[:cnt]
    s1 = c.rigid_contact_shape1.numpy()[:cnt]
    f  = c.force.numpy()[:cnt][:, :3]
    nw = c.rigid_contact_normal.numpy()[:cnt]
    dist = mjd.contact.dist.numpy()[:cnt]
    for k in range(cnt):
        a, b = int(s0[k]), int(s1[k])
        if a not in foot_shapes and b not in foot_shapes: continue
        sgn = 1.0 if a in foot_shapes else -1.0
        F = sgn * f[k]
        n = nw[k]
        nn = float(np.dot(n, n))
        if nn > 1e-12: n = n / np.sqrt(nn)
        fn = float(np.dot(F, n)) * n          # normal component
        ft = F - fn                            # friction (tangential) component
        rows.append((float(F[2]), float(nw[k][2]), float(dist[k]), float(fn[2]), float(ft[2])))

rows = np.array(rows) if rows else np.zeros((0, 5))
print(f"[pen] foot contacts: {len(rows)}", flush=True)
if len(rows):
    fz, nz, d, fnz, ftz = rows[:,0], rows[:,1], rows[:,2], rows[:,3], rows[:,4]
    neg, pos = fz < -1.0, fz > 1.0
    for name, mask in (("NEGATIVE-Fz", neg), ("positive-Fz", pos)):
        if mask.sum() == 0: print(f"[pen] {name}: none"); continue
        dd = d[mask]
        print(f"[pen] {name}: n={mask.sum()}  Fz mean={fz[mask].mean():8.1f}  "
              f"dist(m) min={dd.min():+.4f} median={np.median(dd):+.4f} max={dd.max():+.4f}  "
              f"penetrating(dist<0): {100*(dd<0).mean():.1f}%  deep(<-5mm): {100*(dd<-0.005).mean():.1f}%")
        print(f"[pen]     normal.z: median={np.median(nz[mask]):+.3f}  |nz|<0.1: {100*(np.abs(nz[mask])<0.1).mean():.1f}%")
        print(f"[pen]     vertical force split: from NORMAL component mean={fnz[mask].mean():+8.1f} N   "
              f"from FRICTION component mean={ftz[mask].mean():+8.1f} N   "
              f"friction share of |Fz|: {100*np.abs(ftz[mask]).sum()/max(np.abs(fz[mask]).sum(),1e-9):.1f}%")
    # does force magnitude track penetration depth?
    deep = d < -0.005
    if deep.sum():
        print(f"[pen] contacts deeper than 5 mm: {deep.sum()}  mean Fz={fz[deep].mean():.1f}  "
              f"fraction of those with Fz<-1: {100*(fz[deep] < -1).mean():.1f}%")
    shallow = d >= -0.001
    if shallow.sum():
        print(f"[pen] contacts shallower than 1 mm: {shallow.sum()}  mean Fz={fz[shallow].mean():.1f}  "
              f"fraction with Fz<-1: {100*(fz[shallow] < -1).mean():.1f}%")
env.close(); simulation_app.close()
