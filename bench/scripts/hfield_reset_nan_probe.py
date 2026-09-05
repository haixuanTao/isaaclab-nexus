#!/usr/bin/env python3
"""Where do the NaN joints on the (native) heightfield come from?
Runs the real rough-terrain env with zero actions, hooks ManagerBasedRLEnv._reset_idx and
checks joint/root finiteness (a) right after the reset write and (b) after each env step.
For every non-finite env it prints whether it was just reset, its terrain level/type,
root position relative to the cell origin and the deepest MuJoCo contact of that world."""
import argparse, os, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--envs", type=int, default=512); parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--events", type=int, default=3)
parser.add_argument("--actions", type=str, default="zero", help="zero | random  (random = N(0,1) like an untrained policy)")
parser.add_argument("--label", type=str, default="rn")
AppLauncher.add_app_launcher_args(parser); args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, numpy as np, torch
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_newton.physics import NewtonManager
from agile.rl_env.tasks.stand_up.g1.pre_learn import pre_learn
cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.envs); env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped
from agile.rl_env.tasks.stand_up.g1.agents.rsl_rl_ppo_cfg import G1HeightTrackingPpoRunnerCfg
pre_learn(u, args_cli.task, G1HeightTrackingPpoRunnerCfg())  # sets up the fallen-state dataset the reset event needs
robot = u.scene["robot"]; terrain = u.scene.terrain; s = NewtonManager._solver; mjd = s.mjw_data
T = lambda x: x.torch if hasattr(x, "torch") else x
last_reset = {"ids": torch.zeros(0, dtype=torch.long, device=u.device)}
orig_reset = type(u)._reset_idx
_watch = {"ids": None, "n": 0}
def _nan_worlds():
    st = NewtonManager.get_state_0(); nq = int(wp.to_torch(st.joint_q).numel() // u.num_envs)
    return (~torch.isfinite(wp.to_torch(st.joint_q).view(u.num_envs, nq))).any(1)
def _stage(name, obj, attr):
    orig = getattr(obj, attr)
    def wrapped(*a, **k):
        out = orig(*a, **k)
        if _watch["ids"] is not None and _watch["n"] < 40:
            _watch["n"] += 1
            still = _nan_worlds()[_watch["ids"]]
            print(f"[rn]     after {name:32s}: watched worlds NaN {int(still.sum())}/{len(_watch['ids'])}", flush=True)
        return out
    setattr(obj, attr, wrapped)
for nm, ob, at in (("curriculum.compute", u.curriculum_manager, "compute"), ("scene.reset", u.scene, "reset"), ("event.apply(reset)", u.event_manager, "apply"),
                   ("observation.reset", u.observation_manager, "reset"), ("action.reset", u.action_manager, "reset"), ("reward.reset", u.reward_manager, "reset"),
                   ("curriculum.reset", u.curriculum_manager, "reset"), ("command.reset", u.command_manager, "reset"), ("event.reset", u.event_manager, "reset"),
                   ("termination.reset", u.termination_manager, "reset"), ("recorder.reset", u.recorder_manager, "reset")):
    try: _stage(nm, ob, at)
    except Exception as exc: print(f"[rn] could not wrap {nm}: {exc}")
def _wrap_write(name):
    orig = getattr(type(robot), name)
    def w(self, *a, **k):
        ids = k.get("env_ids", a[1] if len(a) > 1 and name.startswith("write_root") else (a[3] if len(a) > 3 else None))
        watched = _watch["ids"]
        pre = _nan_worlds()[watched] if watched is not None else None
        out = orig(self, *a, **k)
        if watched is not None and _watch["n"] < 60:
            _watch["n"] += 1
            post = _nan_worlds()[watched]
            try:
                idt = torch.as_tensor(ids, device=u.device).long() if ids is not None else None
                cover = int(torch.isin(watched, idt).sum()) if idt is not None else -1
                arg0 = a[0] if a else k.get("position", k.get("root_pose", k.get("root_velocity")))
                nan_in = int((~torch.isfinite(arg0.float())).any(-1).sum()) if torch.is_tensor(arg0) else -1
            except Exception as exc:
                cover, nan_in = f"?{exc}", -1
            print(f"[rn]       {name:28s} env_ids covers {cover}/{len(watched)} watched; NaN rows in arg0={nan_in}; watched NaN before={int(pre.sum())} after={int(post.sum())}", flush=True)
        return out
    setattr(type(robot), name, w)
for _n in ("write_root_pose_to_sim", "write_root_velocity_to_sim", "write_root_state_to_sim", "write_root_link_pose_to_sim", "write_root_com_velocity_to_sim",
           "write_joint_state_to_sim", "write_joint_position_to_sim", "write_joint_velocity_to_sim"):
    if hasattr(type(robot), _n): _wrap_write(_n)
def reset_checked(self, env_ids):
    ids0 = torch.as_tensor(env_ids, device=u.device).long()
    pre = _nan_worlds(); w = ids0[pre[ids0]]
    _watch["ids"] = w if len(w) else None
    if _watch["ids"] is not None and _watch["n"] < 40:
        print(f"[rn]   _reset_idx({len(ids0)} envs): {len(w)} of them are NaN worlds BEFORE the reset (e.g. {w[:4].tolist()})", flush=True)
    orig_reset(self, env_ids)
    _watch["ids"] = None
    ids = torch.as_tensor(env_ids, device=u.device).long(); last_reset["ids"] = ids
    jp = T(robot.data.joint_pos)[ids]; rp = T(robot.data.root_pos_w)[ids]
    bad = ~(torch.isfinite(jp).all(1) & torch.isfinite(rp).all(1))
    if bad.any():
        print(f"[rn] NON-FINITE RIGHT AFTER RESET WRITE in {int(bad.sum())}/{len(ids)} envs: {ids[bad][:8].tolist()}", flush=True)
        e = int(ids[bad][0]); scan_world(e)
        try:
            lim = T(robot.data.joint_pos_limits)[e]; dflt = T(robot.data.default_joint_pos)[e]
            st = NewtonManager.get_state_0(); nq = int(wp.to_torch(st.joint_q).numel() // u.num_envs)
            q_before = wp.to_torch(st.joint_q)[e * nq:(e + 1) * nq]
            print(f"[rn]   env {e}: joint_pos_limits finite={bool(torch.isfinite(lim).all())} default_joint_pos finite={bool(torch.isfinite(dflt).all())} "
                  f"state_0.joint_q[world] non-finite={int((~torch.isfinite(q_before)).sum())}/{nq}", flush=True)
            eid = torch.tensor([e], device=u.device)
            robot.write_joint_state_to_sim(dflt.unsqueeze(0).clone(), torch.zeros_like(dflt).unsqueeze(0), env_ids=eid)
            q_after = wp.to_torch(st.joint_q)[e * nq:(e + 1) * nq]; jp_after = T(robot.data.joint_pos)[e]
            print(f"[rn]   env {e}: after DIRECT default write -> state_0.joint_q non-finite={int((~torch.isfinite(q_after)).sum())}/{nq}, "
                  f"data.joint_pos finite={bool(torch.isfinite(jp_after).all())}", flush=True)
            mjq = wp.to_torch(mjd.qpos)[e] if hasattr(mjd, "qpos") else None
            if mjq is not None: print(f"[rn]   env {e}: mjw_data.qpos[world] non-finite={int((~torch.isfinite(mjq)).sum())}/{mjq.numel()} (synced from state at next step)", flush=True)
        except Exception as exc:
            print(f"[rn]   direct-write test failed: {exc}", flush=True)
type(u)._reset_idx = reset_checked
# ---- bisect inside the reset: are the written joints finite at write time, and after forward()? ----
_wlog = {"n": 0}
_orig_fwd = NewtonManager.forward.__func__
def fwd_checked(cls):
    st = NewtonManager.get_state_0(); nq = int(wp.to_torch(st.joint_q).numel() // u.num_envs)
    before = (~torch.isfinite(wp.to_torch(st.joint_q).view(u.num_envs, nq))).any(1)
    _orig_fwd(cls)
    after = (~torch.isfinite(wp.to_torch(st.joint_q).view(u.num_envs, nq))).any(1)
    new_bad = after & ~before
    if new_bad.any() and _wlog["n"] < 12:
        _wlog["n"] += 1
        print(f"[rn]   NewtonManager.forward() turned {int(new_bad.sum())} finite worlds non-finite (e.g. {torch.where(new_bad)[0][:4].tolist()})", flush=True)
NewtonManager.forward = classmethod(fwd_checked)
import warp as wp
_scanned = {"n": 0}
def scan_world(world, limit=3):
    """List every per-world MuJoCo-Warp data array that is non-finite for `world` (and Newton state_0)."""
    if _scanned["n"] >= limit: return
    _scanned["n"] += 1
    nworld = int(mjd.nworld); bad_fields = []
    for name in dir(mjd):
        if name.startswith("_"): continue
        try: arr = getattr(mjd, name)
        except Exception: continue
        if not isinstance(arr, wp.array) or arr.ndim < 2 or arr.shape[0] != nworld: continue
        if arr.dtype not in (wp.float32, wp.vec3, wp.vec4, wp.quat, wp.mat33, wp.transform, wp.spatial_vector): continue
        try:
            t = wp.to_torch(arr)[world]
            if not torch.isfinite(t.float()).all(): bad_fields.append(f"{name}{tuple(arr.shape)}")
        except Exception: pass
    st = NewtonManager.get_state_0() if hasattr(NewtonManager, "get_state_0") else None
    newton_bad = []
    if st is not None:
        for name in ("joint_q", "joint_qd", "body_q", "body_qd", "joint_f", "body_f"):
            a = getattr(st, name, None)
            if isinstance(a, wp.array):
                t = wp.to_torch(a).float()
                if not torch.isfinite(t).all(): newton_bad.append(f"{name}: {int((~torch.isfinite(t)).sum())} non-finite of {t.numel()}")
    print(f"[rn]   world {world}: non-finite mjw_data per-world arrays: {bad_fields if bad_fields else 'none'}", flush=True)
    print(f"[rn]   newton state_0 non-finite: {newton_bad if newton_bad else 'none'}", flush=True)
def deepest_contact(world):
    n = int(mjd.nacon.numpy()[0])
    if n == 0: return None
    wid = mjd.contact.worldid.numpy()[:n]; m = wid == world
    if not m.any(): return None
    d = mjd.contact.dist.numpy()[:n][m]; return float(d.min()), int(m.sum())
env.reset(); events = 0; act = torch.zeros(u.action_space.shape, device=u.device)
torch.manual_seed(0)
print(f"[rn] actions={args_cli.actions} envs={args_cli.envs} label={args_cli.label}", flush=True)
for step in range(1, args_cli.steps + 1):
    prev_jp = T(robot.data.joint_pos).clone(); prev_rp = T(robot.data.root_pos_w).clone()
    if args_cli.actions == "random": act = torch.randn(u.action_space.shape, device=u.device)
    env.step(act)
    jp = T(robot.data.joint_pos); rp = T(robot.data.root_pos_w); jv = T(robot.data.joint_vel)
    bad = ~(torch.isfinite(jp).all(1) & torch.isfinite(rp).all(1) & torch.isfinite(jv).all(1))
    if step % 50 == 0:
        print(f"[rn] step {step}: resets this step={len(last_reset['ids'])} max|joint_vel|={jv[torch.isfinite(jv)].abs().max().item():.1f}", flush=True)
    if bad.any():
        events += 1; ids = torch.where(bad)[0]
        print(f"\n[rn] EVENT {events} at step {step}: {len(ids)} non-finite envs; of which just reset this step: "
              f"{int(torch.isin(ids, last_reset['ids']).sum())}", flush=True)
        for e in ids[:6].tolist():
            lvl = int(terrain.terrain_levels[e]); typ = int(terrain.terrain_types[e]); org = u.scene.env_origins[e]
            rel = (prev_rp[e] - org).tolist(); dc = deepest_contact(e)
            print(f"[rn]   env {e}: level={lvl} type={typ} root_rel_before=({rel[0]:+.2f},{rel[1]:+.2f},{rel[2]:+.2f}) "
                  f"|joint_pos_before|max={prev_jp[e].abs().max().item():.2f} just_reset={bool((last_reset['ids']==e).any())} "
                  f"deepest contact now={dc}", flush=True)
        if events >= args_cli.events: break
        u._reset_idx(ids)  # clear and continue
print(f"[rn] done: {events} events in {step} steps", flush=True)
env.close(); simulation_app.close()
