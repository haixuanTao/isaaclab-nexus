"""AGILE env on Nexus: right after reset (default reset, then fallen-state dataset reset), which bodies are
below the local terrain, by how much, and where the root is. Steps 20 physics steps after each reset to see
whether penetration self-corrects."""
import os, sys, glob, numpy as np
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym, torch
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_nexus.envs import nexusify
TASK = "HeightTracking-G1-v0"; N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
env_cfg.scene.num_envs = N; env_cfg.seed = 7
nexusify(env_cfg, os.environ.get("NEXUS_G1_MJCF", "/workspace/bench/nexus_port/g1_29dof_convex64.xml"), agent_cfg=agent_cfg)
env = gym.make(TASK, cfg=env_cfg); base = env.unwrapped; robot = base.scene.articulations["robot"]; terr = base.scene.terrain.terrain
def report(tag):
    bp = robot.data.body_link_pos_w.torch; hz = terr.heights_at(bp[..., :2].reshape(N, -1, 2)).reshape(N, -1); cl = bp[..., 2] - hz
    low, idx = cl.min(1); names = np.array(robot.body_names)
    print(f"[{tag}] root z mean {robot.data.root_link_pos_w.torch[:,2].mean():.2f} | lowest-body clearance: median {low.median():+.3f} p10 {low.kthvalue(max(1,int(0.1*N))).values:+.3f} min {low.min():+.3f} | envs < -0.05: {(low<-0.05).float().mean()*100:.0f}%  < -0.2: {(low<-0.2).float().mean()*100:.0f}%")
    worst = torch.argsort(low)[:5]
    for e in worst.tolist():
        print(f"    env {e:3d}: {names[idx[e]]:<22} z {bp[e, idx[e], 2]:+.3f} terrain {hz[e, idx[e]]:+.3f} -> {low[e]:+.3f} | root z {robot.data.root_link_pos_w.torch[e,2]:+.3f} root xy {bp[e,0,:2].cpu().numpy().round(2)} | bodies below: {int((cl[e] < -0.02).sum())}/{cl.shape[1]}")
    bad = names[idx[low < -0.05].cpu().numpy()]; u, c = np.unique(bad, return_counts=True)
    if len(u): print("    lowest-body histogram (envs < -0.05):", dict(zip(u.tolist(), c.tolist())))
def steps(k):
    with torch.no_grad():
        for _ in range(k): base.step(torch.zeros(N, base.action_manager.total_action_dim, device=base.device))
base.reset(); report("default reset, t=0")
steps(1); report("default reset, +1 env step"); steps(20); report("default reset, +21 env steps")
pre = gym.spec(TASK).kwargs.get("pre_learn_entry_point")
if pre:
    import importlib; mod, fn = pre.split(":"); getattr(importlib.import_module(mod), fn)(base, TASK, agent_cfg); base.reset()
    for f in sorted(glob.glob("/workspace/WBC-AGILE/fallen_states_cache_nexus/*.pt")):
        D = torch.load(f, map_location="cpu", weights_only=False); st0 = D["states_by_level"][0]; jv = st0["joint_vel"].abs()
        print(f"[collected dataset N={N}] level-0 states {len(jv)} | root_pos_rel z median {st0['root_pos_rel'][:,2].median():.2f} max {st0['root_pos_rel'][:,2].max():.2f} | joint_vel p99 {jv.flatten().kthvalue(int(0.99*jv.numel())).values:.1f} max {jv.max():.1f} | root_lin_vel max {st0['root_lin_vel'].norm(dim=-1).max():.2f}")
    report("dataset reset, t=0"); steps(1); report("dataset reset, +1 env step"); steps(20); report("dataset reset, +21 env steps")
    # where does the reset put the root relative to the surface?
    rz = robot.data.root_link_pos_w.torch[:, 2]; bp0 = robot.data.body_link_pos_w.torch; hz = terr.heights_at(bp0[..., :2].reshape(N, -1, 2)).reshape(N, -1)[:, 0]
    # FK cross-check: does MuJoCo FK of (root pose, joint_pos) agree with the engine's link buffer for the worst envs?
    import mujoco; m = mujoco.MjModel.from_xml_path("/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"); d = mujoco.MjData(m)
    qadr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): int(m.jnt_qposadr[j]) for j in range(m.njnt)}; jidx = [qadr[n] for n in robot.joint_names]
    bp = robot.data.body_link_pos_w.torch; hz = terr.heights_at(bp[..., :2].reshape(N, -1, 2)).reshape(N, -1); low, idx = (bp[..., 2] - hz).min(1)
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"); fid2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    lf = robot.find_bodies("left_ankle_roll_link")[0][0]; rf = robot.find_bodies("right_ankle_roll_link")[0][0]; dq0 = robot.data.default_joint_pos.torch
    for e in torch.argsort(low)[:4].tolist():
        d.qpos[:3] = robot.data.root_link_pos_w.torch[e].cpu().numpy(); d.qpos[3:7] = robot.data.root_link_quat_w.torch[e][[3, 0, 1, 2]].cpu().numpy(); d.qpos[jidx] = robot.data.joint_pos.torch[e].cpu().numpy(); mujoco.mj_forward(m, d)
        print(f"    FK env {e:3d}: feet z engine buffer L {bp[e, lf, 2]:+.3f} R {bp[e, rf, 2]:+.3f} | MuJoCo FK L {d.xpos[fid][2]:+.3f} R {d.xpos[fid2][2]:+.3f} | |joint_pos - default| mean {(robot.data.joint_pos.torch[e]-dq0[e]).abs().mean():.2f} | root quat {robot.data.root_link_quat_w.torch[e].cpu().numpy().round(2)}")
    print(f"[dataset reset] root height above local terrain: median {(rz-hz).median():+.3f} min {(rz-hz).min():+.3f} | envs with |root xy| > 4 m (off tile): {int((robot.data.root_link_pos_w.torch[:, :2].abs().max(1).values > 4).sum())}")
env.close(); app.close()
