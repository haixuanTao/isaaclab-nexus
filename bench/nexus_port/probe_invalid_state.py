"""Which `invalid_state` condition ends the episodes? Roll a checkpoint with AGILE's env (terminations ON) and,
at every step, count envs exceeding each limit; also track the maxima. usage: probe_invalid_state.py <ckpt> [steps] [N]"""
import os, sys, numpy as np
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym, torch
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
from agile.rl_env.rsl_rl import RslRlVecEnvWrapper, make_rsl_rl_runner
from isaaclab_nexus.envs import nexusify
CKPT = sys.argv[1]; STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 300; N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
TASK = "HeightTracking-G1-v0"
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
if os.environ.get("NEXUS_EMP_NORM", "1") != "0": agent_cfg.empirical_normalization = True
env_cfg.scene.num_envs = N; env_cfg.seed = 7
BACKEND = os.environ.get("NEXUS_BACKEND", "nexus")
if BACKEND == "nexus": nexusify(env_cfg, os.environ.get("NEXUS_G1_MJCF", "/workspace/bench/nexus_port/g1_29dof_convex64.xml"), agent_cfg=agent_cfg)
import agile.rl_env.mdp as _mdp
PREV_Q = None; FIRED = []; WHIST = []; JHIST = []
import isaaclab.utils.math as mu
REASON = {"nan": 0, "joint_vel": 0, "root_height": 0, "root_xy": 0, "lin_vel": 0, "ang_vel": 0, "any": 0}
def _invalid_logged(env, asset_cfg, max_joint_vel=100.0, max_root_height=10.0, max_root_xy_distance=200.0, max_lin_vel=50.0, max_ang_vel=100.0):
    r = env.scene[asset_cfg.name]; d = r.data
    nan = torch.isnan(d.joint_pos.torch).any(-1) | torch.isnan(d.joint_vel.torch).any(-1) | torch.isnan(d.root_pos_w.torch).any(-1) | torch.isnan(d.root_lin_vel_w.torch).any(-1) | torch.isnan(d.root_ang_vel_w.torch).any(-1)
    rel = d.root_pos_w.torch - env.scene.env_origins
    c = {"nan": nan, "joint_vel": d.joint_vel.torch.abs().max(-1).values > max_joint_vel, "root_height": rel[:, 2] > max_root_height, "root_xy": rel[:, :2].norm(dim=-1) > max_root_xy_distance,
         "lin_vel": d.root_lin_vel_w.torch.norm(dim=-1) > max_lin_vel, "ang_vel": d.root_ang_vel_w.torch.norm(dim=-1) > max_ang_vel}
    out = torch.zeros_like(nan)
    for k, v in c.items(): REASON[k] += int(v.sum()); out |= v
    REASON["any"] += int(out.sum()); WHIST.append(d.root_ang_vel_w.torch.norm(dim=-1).clone()); JHIST.append(d.joint_vel.torch.abs().max(-1).values.clone())
    global PREV_Q, FIRED
    q = d.root_quat_w.torch
    if PREV_Q is not None and out.any():
        dq = mu.quat_mul(q, mu.quat_inv(PREV_Q)); w_fd = 2 * torch.atan2(dq[:, :3].norm(dim=-1), dq[:, 3].abs()) / float(env.step_dt)
        wv = d.root_ang_vel_w.torch; vv = d.root_lin_vel_w.torch; jv = d.joint_vel.torch.abs().max(-1).values
        bp = d.body_link_pos_w.torch; lowest = bp[..., 2].min(-1).values
        for e in torch.nonzero(out).flatten().tolist()[:3]:
            if len(FIRED) < 12: FIRED.append(f"env {e}: |w| reported {wv[e].norm():.1f} (vec {wv[e].cpu().numpy().round(1)}) FD {w_fd[e]:.1f} | |v| {vv[e].norm():.1f} | max|jv| {jv[e]:.0f} | root z {rel[e,2]:.2f} lowest body z {lowest[e]:.2f}")
    PREV_Q = q.clone(); return out
if os.environ.get("NEXUS_NO_LIFT") == "1":
    env_cfg.actions.lift.stiffness_forces = 0.0; env_cfg.actions.lift.damping_forces = 0.0; env_cfg.actions.lift.damping_torques = 0.0; print("LIFT HARNESS DISABLED")
env_cfg.terminations.invalid_state.func = _invalid_logged                       # same logic as mdp.invalid_state, with counters
env = gym.make(TASK, cfg=env_cfg); base = env.unwrapped; robot = base.scene.articulations["robot"]
pre = gym.spec(TASK).kwargs.get("pre_learn_entry_point")
if pre:
    import importlib; mod, fn = pre.split(":"); getattr(importlib.import_module(mod), fn)(base, TASK, agent_cfg); base.reset()
wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions); runner = make_rsl_rl_runner(wenv, agent_cfg, log_dir=None, device=agent_cfg.device)
runner.load(CKPT); policy = runner.get_inference_policy(device=agent_cfg.device); obs = wenv.get_observations()
P = base_params = dict(env_cfg.terminations.invalid_state.params); lim = {k: P.get(k) for k in ("max_joint_vel", "max_root_height", "max_root_xy_distance", "max_lin_vel", "max_ang_vel")}
print("invalid_state limits:", lim)
cnt = {k: 0 for k in lim}; cnt["nan"] = 0; mx = {k: 0.0 for k in lim}; dones = 0; jmax_hist = []
vl = robot.data.joint_velocity_limits; vl = vl.torch if hasattr(vl, "torch") else vl
print("backend joint_velocity_limits (env 0):", {n: round(float(v), 1) for n, v in zip(robot.joint_names, vl[0])})
import isaaclab.utils.math as mu
q_prev = robot.data.root_quat_w.torch.clone(); fd_worst = []
with torch.inference_mode():
    for i in range(STEPS):
        act = policy(obs)
        if os.environ.get("NEXUS_RANDOM_ACTIONS") == "1": act = torch.randn_like(act)      # iteration-0 policy: N(0,1) actions
        obs, _, done, _ = wenv.step(act); d = robot.data
        jv = d.joint_vel.torch.abs().max(1).values; rz = d.root_pos_w.torch[:, 2] - base.scene.env_origins[:, 2]; xy = (d.root_pos_w.torch[:, :2] - base.scene.env_origins[:, :2]).norm(dim=-1)
        lv = d.root_lin_vel_w.torch.norm(dim=-1); av = d.root_ang_vel_w.torch.norm(dim=-1)
        q = {"max_joint_vel": jv, "max_root_height": rz, "max_root_xy_distance": xy, "max_lin_vel": lv, "max_ang_vel": av}
        for k, v in q.items():
            cnt[k] += int((v > lim[k]).sum()); mx[k] = max(mx[k], float(v.max()))
        cnt["nan"] += int(torch.isnan(d.joint_pos.torch).any(1).sum()); dones += int(done.sum()); jmax_hist.append(float(jv.max()))
        tq = d.applied_torque; tq = (tq.torch if hasattr(tq, "torch") else tq).abs().max(0).values; TQMAX = torch.maximum(TQMAX, tq) if "TQMAX" in dir() else tq.clone()
        q = d.root_quat_w.torch; dq = mu.quat_mul(q, mu.quat_inv(q_prev)); ang = 2.0 * torch.atan2(dq[:, :3].norm(dim=-1), dq[:, 3].abs()); w_fd = ang / float(base.step_dt)
        keep = ~done                                              # skip envs that were just reset (quat jump)
        if keep.any():
            e = int(torch.argmax(av * keep)); fd_worst.append((float(av[e]), float(w_fd[e])))
        q_prev = q.clone()
print(f"[{BACKEND}] random_actions={os.environ.get('NEXUS_RANDOM_ACTIONS')=='1'} invalid_state fired per condition (pre-reset state):", REASON)
_w = torch.cat(WHIST); _j = torch.cat(JHIST); _q = lambda t, p: float(t.kthvalue(max(1, int(p * t.numel()))).values)
print(f"pre-reset root |w| over all env-steps: p50 {_q(_w,.5):.1f} p90 {_q(_w,.9):.1f} p99 {_q(_w,.99):.1f} p99.9 {_q(_w,.999):.1f} max {_w.max():.1f} | frac>25 {(_w>25).float().mean():.4f} >40 {(_w>40).float().mean():.4f} >50 {(_w>50).float().mean():.4f} | max joint |v| p50 {_q(_j,.5):.0f} p99 {_q(_j,.99):.0f}")
print("fired-env details:"); [print("   ", f) for f in FIRED]
print(f"over {STEPS} steps x {N} envs: episodes ended {dones} | env-steps exceeding: " + ", ".join(f"{k} {v}" for k, v in cnt.items()))
top = sorted(fd_worst, reverse=True)[:6]; print("reported root |w| vs finite-difference |w| (top envs, non-reset):", [(round(a,1), round(b,1)) for a, b in top])
el = robot.data.joint_effort_limits; el = el.torch if hasattr(el, "torch") else el
print("max |applied_torque| / effort limit per joint:", {n: f"{float(t):.0f}/{float(l):.0f}" for n, t, l in zip(robot.joint_names, TQMAX, el[0])})
print("maxima: " + ", ".join(f"{k} {v:.1f}" for k, v in mx.items()) + f" | joint_vel max per step p50 {np.median(jmax_hist):.0f} p90 {np.percentile(jmax_hist, 90):.0f}")
tl = ", ".join(f"{n}:{float(v):.0f}" for n, v in zip(robot.joint_names, d.joint_vel.torch.abs().max(0).values)) if hasattr(d, "joint_vel") else ""
print("per-joint |vel| max at the last step:", tl[:400])
env.close(); app.close()
