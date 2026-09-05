#!/usr/bin/env python3
"""Does the heightfield physics surface sit where the terrain mesh is?
For every terrain contact: contact point z (MuJoCo contact.pos) vs the mesh height at
the same (x, y) (Warp ray query on the captured trimesh). A raster row/column or sign
convention error shows up as a large, position-dependent mismatch."""
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--envs", type=int, default=32); parser.add_argument("--steps", type=int, default=80)
AppLauncher.add_app_launcher_args(parser); args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, numpy as np, torch, warp as wp
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_newton.physics import NewtonManager
from agile.isaaclab_extras import newton_heightfield_terrain as hft
cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.envs); env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; env.reset()
m = NewtonManager.get_model(); s = NewtonManager._solver; mjd = s.mjw_data
labels = list(m.shape_label); hf_shapes = {i for i, l in enumerate(labels) if "heightfield" in l.lower()}
print(f"\n[al] heightfield shapes: {sorted(hf_shapes)}   robot shapes: {len(labels) - len(hf_shapes)}", flush=True)
info = hft.CAPTURED["terrain"]; mesh = info["mesh"]; r = info["raster"]
v = np.asarray(mesh.vertices, dtype=np.float32); f = np.asarray(mesh.faces, dtype=np.int32).reshape(-1)
wm = wp.Mesh(points=wp.array(v, dtype=wp.vec3), indices=wp.array(f, dtype=wp.int32))
@wp.kernel
def mesh_h(mesh_id: wp.uint64, pts: wp.array(dtype=wp.vec2), z_top: float, max_t: float, out: wp.array(dtype=wp.float32)):
    i = wp.tid(); q = wp.mesh_query_ray(mesh_id, wp.vec3(pts[i][0], pts[i][1], z_top), wp.vec3(0.0, 0.0, -1.0), max_t)
    out[i] = z_top - q.t if q.result else -99.0
def mesh_height(xy):
    pts = wp.array(xy.astype(np.float32), dtype=wp.vec2); out = wp.zeros(len(xy), dtype=wp.float32)
    wp.launch(mesh_h, dim=len(xy), inputs=[wm.id, pts, float(v[:, 2].max()) + 1.0, float(v[:, 2].max() - v[:, 2].min()) + 2.0], outputs=[out]); return out.numpy()
def raster_height(xy):
    c = np.clip(np.round((xy[:, 0] - r["x_min"]) / r["res"]).astype(int), 0, r["ncol"] - 1); rr = np.clip(np.round((xy[:, 1] - r["y_min"]) / r["res"]).astype(int), 0, r["nrow"] - 1)
    return r["h"][rr, c]
errs, errs_r, xs, ys, deep = [], [], [], [], []
for step in range(args_cli.steps):
    env.step(torch.zeros(u.action_space.shape, device=u.device))
    n = int(mjd.nacon.numpy()[0])
    if n == 0: continue
    geom = mjd.contact.geom.numpy()[:n]; pos = mjd.contact.pos.numpy()[:n]; dist = mjd.contact.dist.numpy()[:n]
    g2s = s.mjc_geom_to_newton_shape.numpy(); wid = mjd.contact.worldid.numpy()[:n]
    for k in range(n):
        sh = {int(g2s[wid[k], geom[k][0]]), int(g2s[wid[k], geom[k][1]])}
        if sh & hf_shapes:
            xs.append(pos[k][0]); ys.append(pos[k][1]); errs.append(pos[k][2]); deep.append(dist[k])
xy = np.stack([np.array(xs), np.array(ys)], -1); cz = np.array(errs); hm = mesh_height(xy); hr = raster_height(xy); d = np.array(deep)
ok = hm > -90
e_mesh = cz[ok] - hm[ok]; e_rast = cz[ok] - hr[ok]
print(f"[al] terrain contacts sampled: {len(cz)}  (mesh hit for {ok.sum()})")
print(f"[al] contact z - MESH height at same xy:   median={np.median(e_mesh):+.4f}  p90|.|={np.percentile(np.abs(e_mesh),90):.4f}  max|.|={np.abs(e_mesh).max():.4f} m")
print(f"[al] contact z - RASTER height at same xy: median={np.median(e_rast):+.4f}  p90|.|={np.percentile(np.abs(e_rast),90):.4f}  max|.|={np.abs(e_rast).max():.4f} m")
# alternative conventions, to diagnose a swap: transposed raster / flipped y
def alt(xy, mode):
    c = np.clip(np.round((xy[:, 0] - r["x_min"]) / r["res"]).astype(int), 0, r["ncol"] - 1); rr = np.clip(np.round((xy[:, 1] - r["y_min"]) / r["res"]).astype(int), 0, r["nrow"] - 1)
    if mode == "flip_y": return r["h"][r["nrow"] - 1 - rr, c]
    if mode == "flip_x": return r["h"][rr, r["ncol"] - 1 - c]
    return r["h"][rr, c]
for mode in ("flip_y", "flip_x"):
    e = cz[ok] - alt(xy, mode)[ok]; print(f"[al]   if the physics used {mode:6s}: median={np.median(e):+.4f}  p90|.|={np.percentile(np.abs(e),90):.4f}")
print(f"[al] penetration (dist) on terrain contacts: median={np.median(d):+.4f}  min={d.min():+.4f}   fraction deeper than 5 cm: {(d < -0.05).mean():.3f}")
print(f"[al] mesh relief in sampled area: z in [{hm[ok].min():+.3f}, {hm[ok].max():+.3f}]   raster z in [{r['h'].min():+.3f}, {r['h'].max():+.3f}]")
env.close(); simulation_app.close()
