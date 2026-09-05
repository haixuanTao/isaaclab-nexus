"""For every recorded fallen state: lowest foot-sole corner (and lowest link origin) relative to the local terrain
height, via MuJoCo FK and the same tile rasterization the backend uses. Nexus cache (MJCF joint order) vs the
PhysX cache (USD breadth-first order)."""
import os, glob, numpy as np, torch, mujoco, trimesh
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab.terrains import TerrainGenerator
from isaaclab_nexus.terrain import _rasterize
env_cfg = load_cfg_from_registry("HeightTracking-G1-v0", "env_cfg_entry_point")
gen = TerrainGenerator(cfg=env_cfg.scene.terrain.terrain_generator, device="cpu"); mesh = gen.terrain_mesh
V = np.asarray(mesh.vertices, np.float32); F = np.asarray(mesh.faces, np.int64); origins = np.asarray(gen.terrain_origins); sx, sy = env_cfg.scene.terrain.terrain_generator.size
zmin, zmax = float(V[:, 2].min()), float(V[:, 2].max()); RES = 0.05; grids = {}
def grid(r, c):
    if (r, c) not in grids:
        o = origins[r, c]; lo, hi = o[:2] - np.array([sx, sy]) / 2, o[:2] + np.array([sx, sy]) / 2
        inside = np.all((V[:, :2] >= lo - 1e-6) & (V[:, :2] <= hi + 1e-6), axis=1); keep = inside[F].all(axis=1); Fk = F[keep]; used = np.unique(Fk)
        remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used)); Vt = V[used].copy(); Vt[:, :2] -= o[:2]
        xs, ys, Z = _rasterize(trimesh.Trimesh(Vt, remap[Fk], process=False), sx, sy, RES, zmin, zmax); grids[(r, c)] = (Z, float(xs[0]), float(ys[0]))
    return grids[(r, c)]
def h_at(r, c, xy):
    Z, x0, y0 = grid(r, c); i = np.clip(np.round((xy[:, 0] - x0) / RES).astype(int), 0, Z.shape[0] - 1); j = np.clip(np.round((xy[:, 1] - y0) / RES).astype(int), 0, Z.shape[1] - 1); return Z[i, j]
m = mujoco.MjModel.from_xml_path("/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"); d = mujoco.MjData(m)
jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
qadr = {n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in jn}
bfs, q = [], [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]
while q:
    b = q.pop(0); q += [c for c in range(m.nbody) if m.body_parentid[c] == b and c != b]
    bfs += [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt) if m.jnt_bodyid[j] == b and m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
PTS = np.array([[-0.05, 0.025, -0.03], [-0.05, -0.025, -0.03], [0.12, 0.03, -0.03], [0.12, -0.03, -0.03]])
def analyse(f, order, tag):
    D = torch.load(f, map_location="cpu", weights_only=False); sole, link = [], []
    for lv, s in D["states_by_level"].items():
        rp, rq, jp, tt = s["root_pos_rel"].numpy(), s["root_quat"].numpy(), s["joint_pos"].numpy(), s["terrain_type"].numpy()
        for i in range(len(rp)):
            d.qpos[:] = 0; d.qpos[:3] = rp[i]; d.qpos[3:7] = rq[i][[3, 0, 1, 2]]
            for n, v in zip(order, jp[i]): d.qpos[qadr[n]] = v
            mujoco.mj_forward(m, d); r, c = int(lv), int(tt[i])
            corners = np.concatenate([d.xpos[b] + d.xmat[b].reshape(3, 3) @ PTS.T.T[:, :, None].squeeze(-1).T if False else (d.xpos[b][None] + (d.xmat[b].reshape(3, 3) @ PTS.T).T) for b in feet])
            sole.append((corners[:, 2] - 0.005 - h_at(r, c, corners[:, :2]) - 0.0).min())          # recorded height, before AGILE's +0.05 reset offset
            link.append((d.xpos[1:, 2] - h_at(r, c, d.xpos[1:, :2])).min())
    sole, link = np.array(sole), np.array(link)
    print(f"[{tag}] {len(sole)} states | sole corner vs terrain: median {np.median(sole):+.3f} p10 {np.percentile(sole,10):+.3f} min {sole.min():+.3f} | frac < -0.02: {(sole<-0.02).mean():.2f}  < -0.05: {(sole<-0.05).mean():.2f}  < -0.10: {(sole<-0.10).mean():.2f} | lowest link origin p10 {np.percentile(link,10):+.3f}")
for f, order, tag in [(sorted(glob.glob("/workspace/WBC-AGILE/fallen_states_cache_nexus/*8cd61f3f.pt"))[0], jn, "Nexus primary (v16)"), (sorted(glob.glob("/workspace/WBC-AGILE/fallen_states_cache_nexus/*edd3202a.pt"))[0], jn, "Nexus secondary (v16)"),
                      ("/workspace/WBC-AGILE/fallen_states_cache/fallen_states_v6_HeightTracking_G1_v0_548cc4ff_8cd61f3f.pt", bfs, "PhysX primary"), ("/workspace/WBC-AGILE/fallen_states_cache/fallen_states_v6_HeightTracking_G1_v0_548cc4ff_edd3202a.pt", bfs, "PhysX secondary")]:
    analyse(f, order, tag)
app.close()
