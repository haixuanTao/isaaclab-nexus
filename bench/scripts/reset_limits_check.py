#!/usr/bin/env python3
"""After a REAL env.reset() (no writes of my own), how many joints sit outside
their limits, read by NAME? A scrambled index-ordered write shows up here."""
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--num_envs", type=int, default=256)
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
env.reset()
# force a full reset of every env through the real reset path (incl. fallen dataset)
u.reset(env_ids=torch.arange(u.num_envs, device=u.device)) if hasattr(u, "reset") else None
jp = tt(robot.data.joint_pos); lim = tt(robot.data.joint_pos_limits)   # (E,J,2)
lo, hi = lim[..., 0], lim[..., 1]
over = (jp < lo - 0.02) | (jp > hi + 0.02)
names = robot.joint_names
print(f"\n[lim] ===== {args_cli.label} =====  envs={u.num_envs}")
print(f"[lim] (env,joint) pairs outside limits right after reset: {int(over.sum())} of {over.numel()} "
      f"({100*float(over.float().mean()):.2f}%)   envs with >=1 violation: {int(over.any(1).sum())}")
per_joint = over.float().mean(0)
for i in torch.argsort(per_joint, descending=True)[:8].tolist():
    if per_joint[i] > 0:
        print(f"[lim]   {names[i]:26s} violated in {100*float(per_joint[i]):5.1f}% of envs   "
              f"limits [{float(lo[0,i]):+.3f},{float(hi[0,i]):+.3f}]  "
              f"seen range [{float(jp[:,i].min()):+.3f},{float(jp[:,i].max()):+.3f}]")
print(f"[lim] sample named readbacks env0: " + ", ".join(
    f"{n}={float(jp[0,names.index(n)]):+.2f}" for n in
    ["waist_yaw_joint","waist_roll_joint","left_ankle_roll_joint","left_elbow_joint","left_wrist_yaw_joint"]))
env.close(); simulation_app.close()
