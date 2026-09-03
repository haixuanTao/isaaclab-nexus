"""Roll the trained policy on the Nexus backend and record env poses for offline rendering.
usage: record_nexus_policy.py <checkpoint.pt> [steps] [num_envs]
Writes bench/video/nexus_rollout.npz: per-step root pose + joint angles for the first 4 envs,
joint names, and env 0's terrain tile mesh (tile-local XY, world Z — the frame the poses are in)."""
import os, sys, numpy as np
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym, torch
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
from agile.rl_env.rsl_rl import RslRlVecEnvWrapper, make_rsl_rl_runner
from isaaclab_nexus.envs import nexusify
CKPT = sys.argv[1]; STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 400; N = int(sys.argv[3]) if len(sys.argv) > 3 else 64
TASK = "HeightTracking-G1-v0"; G1 = os.environ.get("NEXUS_G1_MJCF", "/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml")
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
if os.environ.get("NEXUS_EMP_NORM", "1") != "0": agent_cfg.empirical_normalization = True   # match train_nexus.py (checkpoints carry normalizer buffers)
env_cfg.scene.num_envs = N; env_cfg.seed = 7
nexusify(env_cfg, G1, solver_iterations=int(os.environ.get('NEXUS_SOLVER_ITERS', '1')), collisions_capacity=int(os.environ.get('NEXUS_COLLISIONS_CAPACITY', '256')))
env = gym.make(TASK, cfg=env_cfg); base = env.unwrapped
robot = base.scene.articulations["robot"]; terr = base.scene.terrain.terrain
pre = gym.spec(TASK).kwargs.get("pre_learn_entry_point")          # fallen-state dataset, as in training
if pre and os.environ.get("NEXUS_RECORD_PRELEARN", "1") == "1":
    import importlib; mod, fn = pre.split(":"); getattr(importlib.import_module(mod), fn)(base, TASK, agent_cfg); base.reset()
wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = make_rsl_rl_runner(wenv, agent_cfg, log_dir=None, device=agent_cfg.device)
runner.load(CKPT); policy = runner.get_inference_policy(device=agent_cfg.device)
obs = wenv.get_observations()
K = 4; root_pos, root_quat, jpos, zall, ext, clear, xyall = [], [], [], [], [], [], []
with torch.inference_mode():
    for i in range(STEPS):
        act = policy(obs) * (0.0 if os.environ.get('NEXUS_ZERO_ACTIONS') == '1' else 1.0)
        obs, _, _, _ = wenv.step(act)
        d = robot.data
        root_pos.append(d.root_link_pos_w.torch[:K].cpu().numpy()); root_quat.append(d.root_link_quat_w.torch[:K][:, [3, 0, 1, 2]].cpu().numpy())   # (x,y,z,w) -> MuJoCo (w,x,y,z)
        jpos.append(d.joint_pos.torch[:K].cpu().numpy()); zall.append(d.root_link_pos_w.torch[:, 2].cpu().numpy()); xyall.append(d.root_link_pos_w.torch[:, :2].cpu().numpy())
        jv = d.joint_vel.torch.abs().max().item(); rv = d.root_lin_vel_w.torch.norm(dim=-1).max().item() if hasattr(d, 'root_lin_vel_w') else float('nan')
        ext.append((jv, rv))
        bp = d.body_link_pos_w.torch                                              # (N, B, 3)
        hz = terr.heights_at(bp[..., :2].reshape(bp.shape[0], -1, 2)).reshape(bp.shape[0], bp.shape[1])   # terrain height under every body
        clear.append((bp[..., 2] - hz).min(1).values.cpu().numpy())               # per env: lowest body above local terrain (m)
tiles = [tuple(x) for x in np.asarray([[int(r), int(c)] for r, c in zip(base.scene.terrain.terrain_levels.tolist(), base.scene.terrain.terrain_types.tolist())])]
V, F = terr.tile_vertices[tiles[0]], terr.tile_faces[tiles[0]]
tile_meshes = {f"terrain_v{k}": np.asarray(terr.tile_vertices[tiles[k]], np.float32) for k in range(K)}
tile_meshes.update({f"terrain_f{k}": np.asarray(terr.tile_faces[tiles[k]], np.int32) for k in range(K)})
cl = np.stack(clear)                                                                # (T, N)
print(f"body-terrain clearance (lowest body vs local terrain, m): median over (t,env) {np.median(cl):+.3f} | p1 {np.percentile(cl,1):+.3f} | min {cl.min():+.3f} | fraction of (t,env) below -0.05 m: {(cl < -0.05).mean():.3f}")
xy = np.stack(xyall); off = np.abs(xy).max(-1) > 4.0                                        # (T, N) root off its 8x8 m tile
print("clearance by location: on-tile samples " + f"{(~off).mean():.2f} of all | on-tile: median {np.median(cl[~off]):+.3f} p1 {np.percentile(cl[~off],1):+.3f} frac<-0.05 {(cl[~off]<-0.05).mean():.3f} frac<-0.2 {(cl[~off]<-0.2).mean():.3f}" + (f" | off-tile: median {np.median(cl[off]):+.3f} p1 {np.percentile(cl[off],1):+.3f} frac<-0.2 {(cl[off]<-0.2).mean():.3f}" if off.any() else " | no off-tile samples"))
print("on-tile time course (frac of envs with a body < -0.05 / < -0.2 m): " + " | ".join(f"t={t}s {((cl[min(int(t/float(base.step_dt)),len(cl)-1)] < -0.05) & ~off[min(int(t/float(base.step_dt)),len(cl)-1)]).mean():.2f}/{((cl[min(int(t/float(base.step_dt)),len(cl)-1)] < -0.2) & ~off[min(int(t/float(base.step_dt)),len(cl)-1)]).mean():.2f}" for t in (0.1, 1, 2, 4, 8)))
np.savez("/workspace/bench/video/nexus_rollout.npz", root_pos=np.stack(root_pos), root_quat=np.stack(root_quat), joint_pos=np.stack(jpos),
         joint_names=np.array(robot.joint_names), root_z_all=np.stack(zall), terrain_v=np.asarray(V, np.float32), terrain_f=np.asarray(F, np.int32), dt=float(base.step_dt), clearance=cl, root_xy_all=xy, **tile_meshes)
za = np.stack(zall); t8 = min(int(8 / float(base.step_dt)), len(za) - 1)
import numpy as _np; e=_np.array(ext); print(f"EXTREMES over {len(e)} steps x {za.shape[1]} envs: max |joint_vel| {e[:,0].max():.1f} rad/s (p99 over steps {_np.percentile(e[:,0],99):.1f}) | max |root_vel| {_np.nanmax(e[:,1]):.2f} m/s | min z {za.min():.2f} max z {za.max():.2f} | solver_iters {os.environ.get('NEXUS_SOLVER_ITERS','1')}")
print(f"ALL {za.shape[1]} envs: root z mean at t=0/2/4/8 s = " + " / ".join(f"{za[min(int(t/float(base.step_dt)),len(za)-1)].mean():.2f}" for t in (0,2,4,8))
      + f" | fraction with z>0.5 at t=8s: {(za[t8] > 0.5).mean():.2f} | max z at t=8s: {za[t8].max():.2f}")
from isaaclab_nexus.physics.nexus_manager import NexusManager as _NM; print("rbd_resize_stats:", _NM._state.rbd_resize_stats() if hasattr(_NM._state, "rbd_resize_stats") else "n/a", "| capacity requested", os.environ.get("NEXUS_COLLISIONS_CAPACITY", "256"))
print(f"recorded {STEPS} steps x {K} envs -> bench/video/nexus_rollout.npz | terrain tile {tiles[0]} {len(F)} faces | quat order (w,x,y,z)")
env.close(); app.close()
