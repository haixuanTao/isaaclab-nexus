#!/usr/bin/env python3
"""Does the heightfield physics surface sit where the terrain mesh is?
For every terrain contact: contact point z (MuJoCo contact.pos) vs the mesh height at
the same (x, y) (Warp ray query on the captured trimesh). A raster row/column or sign
convention error shows up as a large, position-dependent mismatch."""
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--envs", type=int, default=32); parser.add_argument("--steps", type=int, default=80)
parser.add_argument("--spread_levels", action="store_true", help="spread envs over all terrain rows before sampling")
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
try:  # native (Isaac Lab develop) heightfields carry the terrain prim label -> find them by geometry type
    from newton import GeoType
    hf_shapes |= {int(i) for i in np.where(m.shape_type.numpy() == int(GeoType.HFIELD))[0]}
except Exception as exc:
    print(f"[al] shape_type scan skipped: {exc}")
print(f"\n[al] heightfield shapes: {sorted(hf_shapes)}   robot shapes: {len(labels) - len(hf_shapes)}", flush=True)
if args_cli.spread_levels and getattr(u.scene.terrain, "terrain_levels", None) is not None:
    nlv = u.scene.terrain.terrain_origins.shape[0]
    u.scene.terrain.terrain_levels[:] = torch.arange(u.num_envs, device=u.device) % nlv
    env.reset(); print(f"[al] envs spread over {nlv} terrain rows", flush=True)
info = hft.CAPTURED.get("terrain") or {"mesh": next(iter(u.scene.terrain.meshes.values()))}  # native path (develop): wrapper off, take the importer's trimesh
mesh = info["mesh"]; r = info.get("raster")  # None on Isaac Lab develop (native conversion, no wrapper raster)
v = np.asarray(mesh.vertices, dtype=np.float32); f = np.asarray(mesh.faces, dtype=np.int32).reshape(-1)
# ---- what does the physics actually hold for the heightfield? ----
try:
    import mujoco
    for sh in sorted(hf_shapes):
        src = m.shape_source[sh]; tf = m.shape_transform.numpy()[sh]
        print(f"[al] newton hfield shape {sh}: nrow={src.nrow} ncol={src.ncol} hx={src.hx:.2f} hy={src.hy:.2f} min_z={src.min_z:+.4f} max_z={src.max_z:+.4f} "
              f"data[0,1] range=[{float(src.data.min()):.3f},{float(src.data.max()):.3f}] xform p=({tf[0]:+.2f},{tf[1]:+.2f},{tf[2]:+.4f})", flush=True)
    mj = s.mj_model
    for g in range(mj.ngeom):
        if mj.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD:
            h = mj.geom_dataid[g]; sz = mj.hfield_size[h]; adr = mj.hfield_adr[h]; n_ = mj.hfield_nrow[h] * mj.hfield_ncol[h]
            dat = mj.hfield_data[adr:adr + n_]
            print(f"[al] mujoco hfield geom {g}: pos=({mj.geom_pos[g][0]:+.2f},{mj.geom_pos[g][1]:+.2f},{mj.geom_pos[g][2]:+.4f}) size(hx,hy,z_top,z_base)=({sz[0]:.2f},{sz[1]:.2f},{sz[2]:.4f},{sz[3]:.4f}) "
                  f"data range=[{dat.min():.3f},{dat.max():.3f}] -> surface z in [{mj.geom_pos[g][2] + dat.min()*sz[2]:+.4f},{mj.geom_pos[g][2] + dat.max()*sz[2]:+.4f}]", flush=True)
    # and the per-world geom_pos the solver actually uses (mjw_model), world 0
    gp = s.mjw_model.geom_pos.numpy(); hg = [g for g in range(mj.ngeom) if mj.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD]
    if hg: print(f"[al] mjw_model.geom_pos[world0, hfield] = {gp[0, hg[0]] if gp.ndim == 3 else gp[hg[0]]}", flush=True)
    print(f"[al] terrain mesh z range: [{float(v[:, 2].min()):+.4f}, {float(v[:, 2].max()):+.4f}]", flush=True)
except Exception as exc:
    print(f"[al] hfield parameter dump failed: {exc}", flush=True)
wm = wp.Mesh(points=wp.array(v, dtype=wp.vec3), indices=wp.array(f, dtype=wp.int32))
@wp.kernel
def mesh_h(mesh_id: wp.uint64, pts: wp.array(dtype=wp.vec2), z_top: float, max_t: float, out: wp.array(dtype=wp.float32)):
    i = wp.tid(); q = wp.mesh_query_ray(mesh_id, wp.vec3(pts[i][0], pts[i][1], z_top), wp.vec3(0.0, 0.0, -1.0), max_t)
    out[i] = z_top - q.t if q.result else -99.0
def mesh_height(xy):
    pts = wp.array(xy.astype(np.float32), dtype=wp.vec2); out = wp.zeros(len(xy), dtype=wp.float32)
    wp.launch(mesh_h, dim=len(xy), inputs=[wm.id, pts, float(v[:, 2].max()) + 1.0, float(v[:, 2].max() - v[:, 2].min()) + 2.0], outputs=[out]); return out.numpy()
def raster_height(xy):
    if r is None: return np.full(len(xy), np.nan, dtype=np.float32)
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
if r is not None: print(f"[al] contact z - RASTER height at same xy: median={np.median(e_rast):+.4f}  p90|.|={np.percentile(np.abs(e_rast),90):.4f}  max|.|={np.abs(e_rast).max():.4f} m")
# alternative conventions, to diagnose a swap: transposed raster / flipped y
def alt(xy, mode):
    c = np.clip(np.round((xy[:, 0] - r["x_min"]) / r["res"]).astype(int), 0, r["ncol"] - 1); rr = np.clip(np.round((xy[:, 1] - r["y_min"]) / r["res"]).astype(int), 0, r["nrow"] - 1)
    if mode == "flip_y": return r["h"][r["nrow"] - 1 - rr, c]
    if mode == "flip_x": return r["h"][rr, r["ncol"] - 1 - c]
    return r["h"][rr, c]
for mode in (("flip_y", "flip_x") if r is not None else ()):
    e = cz[ok] - alt(xy, mode)[ok]; print(f"[al]   if the physics used {mode:6s}: median={np.median(e):+.4f}  p90|.|={np.percentile(np.abs(e),90):.4f}")
print(f"[al] penetration (dist) on terrain contacts: median={np.median(d):+.4f}  min={d.min():+.4f}   fraction deeper than 5 cm: {(d < -0.05).mean():.3f}")
print(f"[al] mesh relief in sampled area: z in [{hm[ok].min():+.3f}, {hm[ok].max():+.3f}]" + (f"   raster z in [{r['h'].min():+.3f}, {r['h'].max():+.3f}]" if r is not None else ""))
# ---- per terrain cell: where are the misaligned / deep contacts? ----
try:
    tor = u.scene.terrain.terrain_origins  # (rows=levels, cols=types, 3)
    R_, C_ = tor.shape[0], tor.shape[1]; cells = tor.reshape(-1, 3)[:, :2].cpu().numpy()
    d2 = ((xy[:, None, :] - cells[None, :, :]) ** 2).sum(-1); cid = d2.argmin(1)
    rows_, cols_ = cid // C_, cid % C_
    print(f"[al] per-cell stats over {R_}x{C_} cells (row=level, col=type); contacts assigned to nearest cell centre:")
    worst = []
    for r_ in range(R_):
        for c_ in range(C_):
            m = (rows_ == r_) & (cols_ == c_) & ok
            if m.sum() < 5: continue
            worst.append((float(np.abs(e_mesh[m[ok]] if False else (cz[m] - hm[m])).max()), float(d[m].min()), int(m.sum()), r_, c_))
    worst.sort(reverse=True)
    for e_, dmin_, n_, r_, c_ in worst[:8]:
        print(f"[al]   cell row {r_} col {c_}: n={n_:5d}  max|contact z - mesh z|={e_:.3f} m  min dist={dmin_:+.3f} m")
    per_ring = {"perimeter": [], "interior": []}
    for e_, dmin_, n_, r_, c_ in worst:
        key = "perimeter" if (r_ in (0, R_ - 1) or c_ in (0, C_ - 1)) else "interior"
        per_ring[key].append((e_, dmin_))
    for k, v in per_ring.items():
        if v: print(f"[al]   {k:9s}: cells={len(v)}  median max-error={np.median([a for a, _ in v]):.3f} m  worst min dist={min(b for _, b in v):+.3f} m")
except Exception as exc:
    print(f"[al] per-cell report failed: {exc}")
env.close(); simulation_app.close()
