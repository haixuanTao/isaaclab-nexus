"""Same body-terrain clearance metric on the STOCK PhysX env: per step, per env, the lowest body's height above the
local terrain (tile height grid rasterized exactly as the Nexus backend does it). NEXUS_ZERO_ACTIONS=1 for zero
actions, else the given checkpoint's policy. usage: probe_clearance_physx.py <ckpt> [steps] [num_envs]"""
import os, sys, numpy as np
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym, torch, trimesh
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
from agile.rl_env.rsl_rl import RslRlVecEnvWrapper, make_rsl_rl_runner
from isaaclab_nexus.terrain import _rasterize
CKPT = sys.argv[1]; STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 400; N = int(sys.argv[3]) if len(sys.argv) > 3 else 64
TASK = "HeightTracking-G1-v0"
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
env_cfg.scene.num_envs = N; env_cfg.seed = 7
env = gym.make(TASK, cfg=env_cfg); base = env.unwrapped; robot = base.scene.articulations["robot"]; T = base.scene.terrain
from isaaclab.terrains import TerrainGenerator
gen = TerrainGenerator(cfg=env_cfg.scene.terrain.terrain_generator, device=base.device)   # same cfg + seed as the importer -> identical terrain
mesh = gen.terrain_mesh; V = np.asarray(mesh.vertices, np.float32); F = np.asarray(mesh.faces, np.int64)
origins = np.asarray(gen.terrain_origins); sx, sy = env_cfg.scene.terrain.terrain_generator.size
zmin, zmax = float(V[:, 2].min()), float(V[:, 2].max()); RES = 0.05; grids = {}
def grid(r, c):
    if (r, c) not in grids:
        o = origins[r, c]; lo, hi = o[:2] - np.array([sx, sy]) / 2, o[:2] + np.array([sx, sy]) / 2
        inside = np.all((V[:, :2] >= lo - 1e-6) & (V[:, :2] <= hi + 1e-6), axis=1); keep = inside[F].all(axis=1); Fk = F[keep]; used = np.unique(Fk)
        remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used)); Vt = V[used].copy(); Vt[:, :2] -= o[:2]
        xs, ys, Z = _rasterize(trimesh.Trimesh(Vt, remap[Fk], process=False), sx, sy, RES, zmin, zmax)
        grids[(r, c)] = (torch.as_tensor(Z, device=base.device), float(xs[0]), float(ys[0]), torch.as_tensor(o[:2], device=base.device))
    return grids[(r, c)]

_FOOT_PTS = torch.tensor([[-0.05, 0.025, -0.03], [-0.05, -0.025, -0.03], [0.12, 0.03, -0.03], [0.12, -0.03, -0.03]])
def _sole_clearance(robot, heights_at_fn):
    """min over both feet's 4 sole corners of (corner z - 0.005 - local terrain height), per env."""
    import isaaclab.utils.math as _mu
    ids = robot.find_bodies(".*_ankle_roll_link")[0]; bp = robot.data.body_link_pos_w; bq = robot.data.body_link_quat_w
    bp = (bp.torch if hasattr(bp, "torch") else bp)[:, ids]; bq = (bq.torch if hasattr(bq, "torch") else bq)[:, ids]      # (N, 2, 3/4)
    N = bp.shape[0]; pts = _FOOT_PTS.to(bp.device)
    corners = bp[:, :, None, :] + _mu.quat_apply(bq[:, :, None, :].expand(N, 2, 4, 4).reshape(-1, 4), pts[None, None].expand(N, 2, 4, 3).reshape(-1, 3)).reshape(N, 2, 4, 3)
    c = corners.reshape(N, 8, 3); hz = heights_at_fn(c[..., :2]); return (c[..., 2] - 0.005 - hz).min(1).values

def _heights_world(xy):
    N = xy.shape[0]; out = torch.empty(N, xy.shape[1], device=base.device); lv, ty = T.terrain_levels.tolist(), T.terrain_types.tolist()
    for e in range(N):
        Z, x0, y0, oxy = grid(int(lv[e]), int(ty[e])); q = xy[e] - oxy
        i = ((q[:, 0] - x0) / RES).round().long().clamp(0, Z.shape[0] - 1); j = ((q[:, 1] - y0) / RES).round().long().clamp(0, Z.shape[1] - 1); out[e] = Z[i, j]
    return out
def clearance():
    bp = robot.data.body_link_pos_w.torch if hasattr(robot.data.body_link_pos_w, "torch") else robot.data.body_link_pos_w
    lv, ty = T.terrain_levels.tolist(), T.terrain_types.tolist(); out = torch.empty(N, device=base.device)
    for e in range(N):
        Z, x0, y0, oxy = grid(int(lv[e]), int(ty[e])); xy = bp[e, :, :2] - oxy
        i = ((xy[:, 0] - x0) / RES).round().long().clamp(0, Z.shape[0] - 1); j = ((xy[:, 1] - y0) / RES).round().long().clamp(0, Z.shape[1] - 1)
        out[e] = (bp[e, :, 2] - Z[i, j]).min()
    return out.cpu().numpy()
pre = gym.spec(TASK).kwargs.get("pre_learn_entry_point")
if pre:
    import importlib; mod, fn = pre.split(":"); getattr(importlib.import_module(mod), fn)(base, TASK, agent_cfg); base.reset()
wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions); runner = make_rsl_rl_runner(wenv, agent_cfg, log_dir=None, device=agent_cfg.device)
runner.load(CKPT); policy = runner.get_inference_policy(device=agent_cfg.device); obs = wenv.get_observations(); cl = []; SOLE = []
with torch.inference_mode():
    for i in range(STEPS):
        act = policy(obs) * (0.0 if os.environ.get("NEXUS_ZERO_ACTIONS") == "1" else 1.0); obs, _, _, _ = wenv.step(act); cl.append(clearance()); SOLE.append(_sole_clearance(robot, _heights_world).cpu().numpy())
cl = np.stack(cl); dt = float(base.step_dt); f = lambda t: min(int(t / dt), len(cl) - 1)
print(f"[PhysX, zero_actions={os.environ.get('NEXUS_ZERO_ACTIONS')=='1'}] clearance median over (t,env) {np.median(cl):+.3f} | p1 {np.percentile(cl,1):+.3f} | min {cl.min():+.3f} | frac<-0.05 {(cl<-0.05).mean():.3f} frac<-0.2 {(cl<-0.2).mean():.3f}")
S = np.stack(SOLE); print("SOLE clearance (lowest foot corner vs terrain, m): " + " | ".join(f"t={t}s median {np.median(S[f(t)]):+.3f} p10 {np.percentile(S[f(t)],10):+.3f} frac<-0.02 {(S[f(t)]<-0.02).mean():.2f} frac<-0.05 {(S[f(t)]<-0.05).mean():.2f}" for t in (0.1, 0.5, 1, 4, 8)))
print("time course (frac of envs with a body < -0.05 / < -0.2 m): " + " | ".join(f"t={t}s {(cl[f(t)]<-0.05).mean():.2f}/{(cl[f(t)]<-0.2).mean():.2f}" for t in (0.1, 1, 2, 4, 8)))
env.close(); app.close()
