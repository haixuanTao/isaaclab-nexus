#!/usr/bin/env python3
"""The REAL training reset path: load the fallen-state dataset via the task's
pre_learn hook (exactly as train.py does), reset, and read joints BY NAME.
On Newton the cache is PhysX-ordered and written by index -> limit violations."""
import argparse, sys, importlib
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--label", type=str, default="engine")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, torch
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
def tt(x): return x.torch if hasattr(x, "torch") else x

cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs); cfg.seed = 42
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; robot = u.scene["robot"]

# --- same as train.py: build agent cfg, call pre_learn (loads / generates the dataset) ---
spec = gym.spec(args_cli.task)
mod_name, cls_name = spec.kwargs["rsl_rl_cfg_entry_point"].split(":")
agent_cfg = getattr(importlib.import_module(mod_name), cls_name)()
mod_name, fn_name = spec.kwargs["pre_learn_entry_point"].split(":")
getattr(importlib.import_module(mod_name), fn_name)(u, args_cli.task, agent_cfg)
print(f"[ds] pre_learn hook called (dataset loaded as in training)")

env.reset()
# a couple of full resets so every env has gone through reset_from_fallen_dataset
for _ in range(2):
    u.reset(env_ids=torch.arange(u.num_envs, device=u.device))
    u.scene.write_data_to_sim(); u.sim.step(); u.scene.update(u.physics_dt)

names = robot.joint_names
jp = tt(robot.data.joint_pos); lim = tt(robot.data.joint_pos_limits); lo, hi = lim[..., 0], lim[..., 1]
over = (jp < lo - 0.02) | (jp > hi + 0.02)
print(f"\n[ds] ===== {args_cli.label} =====  envs={u.num_envs}")
print(f"[ds] (env,joint) outside limits after dataset reset: {int(over.sum())} / {over.numel()} "
      f"({100*float(over.float().mean()):.2f}%)   envs with >=1 violation: {int(over.any(1).sum())} / {u.num_envs}")
pj = over.float().mean(0)
for i in torch.argsort(pj, descending=True)[:10].tolist():
    if pj[i] > 0:
        print(f"[ds]   {names[i]:26s} violated {100*float(pj[i]):5.1f}%   limits [{float(lo[0,i]):+.3f},{float(hi[0,i]):+.3f}]"
              f"   seen [{float(jp[:,i].min()):+.3f},{float(jp[:,i].max()):+.3f}]")
env.close(); simulation_app.close()
