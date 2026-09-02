#!/usr/bin/env python3
"""(1) Verify the LOADED dataset by joint NAME on CPU (no reset path involved).
(2) Catch real resets, but refresh sim->data buffers BEFORE reading joints."""
import argparse, sys, importlib
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=800)
parser.add_argument("--label", type=str, default="engine")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, torch
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
def tt(x): return x.torch if hasattr(x, "torch") else x
cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs); cfg.seed = 42
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; robot = u.scene["robot"]
spec = gym.spec(args_cli.task)
mn, cn = spec.kwargs["rsl_rl_cfg_entry_point"].split(":"); agent_cfg = getattr(importlib.import_module(mn), cn)()
mn, fn = spec.kwargs["pre_learn_entry_point"].split(":"); getattr(importlib.import_module(mn), fn)(u, args_cli.task, agent_cfg)
names = robot.joint_names; J = len(names)
lim = tt(robot.data.joint_pos_limits)[0].cpu(); lo, hi = lim[:,0], lim[:,1]

# ---- (1) the loaded dataset itself, by name ----
ds = None
for attr in dir(u):
    v = getattr(u, attr, None)
    if isinstance(getattr(v, "_states_by_level", None), dict) and getattr(v, "_states_by_level"):
        ds = v; break
if ds is None:
    # look on the event term / reset function attributes
    for term in getattr(u.event_manager, "_terms", {}).values() if hasattr(u, "event_manager") else []:
        pass
if ds is None:
    import gc
    for o in gc.get_objects():
        try:
            if type(o).__name__ != "FallenStateDataset":
                continue
            if isinstance(getattr(o, "_states_by_level", None), dict) and o._states_by_level:
                ds = o; break
        except Exception:
            continue
print(f"\n[ds4] ===== {args_cli.label} =====")
if ds is not None:
    jp = torch.cat([s["joint_pos"].cpu() for s in ds._states_by_level.values()])
    dn = list(getattr(ds, "_joint_names", None) or [])
    print(f"[ds4] loaded dataset: {tuple(jp.shape)}  dataset joint order == robot order: {dn == list(names)}")
    over = (jp < lo - 0.02) | (jp > hi + 0.02)
    print(f"[ds4] dataset (sample,joint) outside ROBOT limits by name: {int(over.sum())} / {over.numel()}")
    for n in ["waist_pitch_joint","waist_roll_joint","waist_yaw_joint","left_ankle_roll_joint","left_knee_joint","left_elbow_joint"]:
        i = names.index(n); print(f"[ds4]   {n:24s} mean {float(jp[:,i].mean()):+.3f} absmax {float(jp[:,i].abs().max()):.3f}  limits [{float(lo[i]):+.3f},{float(hi[i]):+.3f}]")
else:
    print("[ds4] could not locate dataset object")

# ---- (2) real resets, buffers refreshed before reading ----
env.reset(); act_dim = u.action_manager.total_action_dim
caught = []
for step in range(args_cli.steps):
    _, _, term, trunc, _ = env.step(torch.randn(u.num_envs, act_dim, device=u.device))
    just = (term | trunc).nonzero().flatten()
    if just.numel():
        u.scene.update(u.physics_dt)            # refresh sim -> data, no physics step
        caught.append(tt(robot.data.joint_pos)[just].cpu().clone())
if caught:
    jp = torch.cat(caught); over = (jp < lo - 0.02) | (jp > hi + 0.02)
    print(f"[ds4] resets caught: {jp.shape[0]}   (reset,joint) outside limits AFTER buffer refresh: {int(over.sum())} / {over.numel()}"
          f"   resets with >=1: {int(over.any(1).sum())}")
    pj = over.float().mean(0)
    for i in torch.argsort(pj, descending=True)[:5].tolist():
        if pj[i] > 0: print(f"[ds4]   {names[i]:24s} violated {100*float(pj[i]):5.1f}%  absmax {float(jp[:,i].abs().max()):.3f}")
env.close(); simulation_app.close()
