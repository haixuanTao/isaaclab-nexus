#!/usr/bin/env python3
"""No teleport, no state writes: env.reset() on the ground, zero action, step.
Track |qd|max and NaN over 100 control steps. Same on both engines."""
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--label", type=str, default="engine")
parser.add_argument("--num_envs", type=int, default=64)
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
env.reset()
print(f"\n[gs] ===== {args_cli.label} =====  actuators={[type(a).__name__ for a in robot.actuators.values()][:1]}")
action = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
line = []
for s in range(100):
    env.step(action)
    qd = tt(robot.data.joint_vel); m = float(qd.abs().max()); fin = bool(torch.isfinite(qd).all())
    if s % 10 == 0 or not fin: line.append(f"s{s}:{m:.2f}{'' if fin else ' NaN'}")
    if not fin: break
print("[gs] |qd|max: " + "  ".join(line))
z = tt(robot.data.root_pos_w)[:, 2]
print(f"[gs] final pelvis z mean={float(z.mean()):.3f}  |qd|max={float(tt(robot.data.joint_vel).abs().max()):.2f}  finite={bool(torch.isfinite(tt(robot.data.joint_vel)).all())}")
env.close(); simulation_app.close()
