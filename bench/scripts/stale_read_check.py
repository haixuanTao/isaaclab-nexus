#!/usr/bin/env python3
"""For envs reset inside env.step(): read joints at that instant, then again one
control step later. If the first read violates limits and the second does not,
the first read was the PRE-reset (terminated) state -- a stale buffer, not the
reset pose."""
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
env.reset(); act_dim = u.action_manager.total_action_dim
lim = tt(robot.data.joint_pos_limits)[0]; lo, hi = lim[:,0], lim[:,1]
def nviol(jp): return int(((jp < lo - 0.02) | (jp > hi + 0.02)).any(1).sum())
at0 = at1 = n = 0; pending = None
for step in range(args_cli.steps):
    _, _, term, trunc, _ = env.step(torch.randn(u.num_envs, act_dim, device=u.device))
    if pending is not None:                       # one step after their reset
        at1 += nviol(tt(robot.data.joint_pos)[pending]); pending = None
    just = (term | trunc).nonzero().flatten()
    if just.numel():
        n += just.numel(); at0 += nviol(tt(robot.data.joint_pos)[just]); pending = just
print(f"\n[stale] ===== {args_cli.label} =====  resets={n}")
print(f"[stale] envs with a joint past its limit  AT the reset step : {at0} / {n}")
print(f"[stale] envs with a joint past its limit  ONE step later    : {at1} / {n}")
print(f"[stale] VERDICT: {'STALE READ at reset step (reset pose is fine)' if at0 > 0 and at1 <= max(2, n//50) else ('reset pose genuinely violates limits' if at1 > n//20 else 'clean')}")
env.close(); simulation_app.close()
