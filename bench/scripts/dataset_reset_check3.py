#!/usr/bin/env python3
"""Exercise the REAL reset path: step with random actions until episodes end,
catch every env at the step it resets (episode_length_buf == 0), and read its
joints BY NAME at that instant. Aggregates over all caught resets."""
import argparse, sys, importlib
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=400)
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
env.reset()
names = robot.joint_names; J = len(names)
lim = tt(robot.data.joint_pos_limits)[0]; lo, hi = lim[:,0], lim[:,1]
dflt = tt(robot.data.default_joint_pos)[0]
act_dim = u.action_manager.total_action_dim
caught = []; n_resets = 0; n_far = 0
for step in range(args_cli.steps):
    env.step(torch.randn(u.num_envs, act_dim, device=u.device))
    just = (u.episode_length_buf == 0).nonzero().flatten()
    if just.numel():
        jp = tt(robot.data.joint_pos)[just]
        caught.append(jp.clone()); n_resets += just.numel()
        n_far += int(((jp - dflt).abs() > 0.3).any(1).sum())
if not caught:
    print(f"[ds3] {args_cli.label}: no resets caught in {args_cli.steps} steps"); sys.exit(0)
jp = torch.cat(caught)
over = (jp < lo - 0.02) | (jp > hi + 0.02)
print(f"\n[ds3] ===== {args_cli.label} =====  resets caught: {n_resets}   of which non-default (dataset/random) poses: {n_far}")
print(f"[ds3] (reset,joint) outside limits: {int(over.sum())} / {over.numel()} ({100*float(over.float().mean()):.2f}%)   "
      f"resets with >=1 violation: {int(over.any(1).sum())} / {n_resets}")
pj = over.float().mean(0)
print(f"[ds3] worst joints by violation rate, with absmax seen vs limit:")
for i in torch.argsort(pj, descending=True)[:8].tolist():
    print(f"[ds3]   {names[i]:26s} violated {100*float(pj[i]):5.1f}%   absmax {float(jp[:,i].abs().max()):.3f}   "
          f"limits [{float(lo[i]):+.3f},{float(hi[i]):+.3f}]")
env.close(); simulation_app.close()
