#!/usr/bin/env python3
"""How often is the joint-velocity limit actually binding, on each engine?

PhysX clamps in-solver, so saturation is measured by counting DOFs sitting at
their limit. Newton has no in-solver clamp; with the AGILE clamp patch active
the same saturation is measured, plus the exact velocity the patch removes.

If saturation is rare, the clamp is a safety net. If it is common, the clamp is
driving the dynamics -- on BOTH engines, since PhysX does the same thing.
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--label", type=str, default="engine")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import agile.isaaclab_extras.monkey_patches  # noqa: F401,E402
import agile.rl_env.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def tt(x):
    return x.torch if hasattr(x, "torch") else x


cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
cfg.seed = 42
env = gym.make(args_cli.task, cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
obs, _ = env.reset()

lim = tt(robot.data.joint_vel_limits).abs().clone()
act_dim = u.action_manager.total_action_dim
frac_hist, near_hist = [], []

for step in range(args_cli.steps):
    obs, *_ = env.step(torch.randn(u.num_envs, act_dim, device=u.device))
    jv = tt(robot.data.joint_vel).abs()
    ok = torch.isfinite(jv)
    if not ok.all():
        print(f"[clamp] non-finite joint_vel at step {step}; stopping", flush=True)
        break
    frac_hist.append(float((jv >= lim * 0.999).float().mean()))
    near_hist.append(float((jv >= lim * 0.95).float().mean()))

n = len(frac_hist)
print(f"\n[clamp] ===== {args_cli.label} =====")
print(f"[clamp] envs={u.num_envs} joints={lim.shape[1]} steps measured={n}")
if n:
    import statistics as st
    print(f"[clamp] DOFs AT the velocity limit (>=99.9%): mean {100*st.mean(frac_hist):6.3f}%  "
          f"max {100*max(frac_hist):6.3f}%")
    print(f"[clamp] DOFs NEAR the limit      (>=95%)   : mean {100*st.mean(near_hist):6.3f}%  "
          f"max {100*max(near_hist):6.3f}%")

try:
    from agile.isaaclab_extras.newton_joint_velocity_clamp import get_clamp_stats
    s = get_clamp_stats()
    if s["steps"]:
        print(f"[clamp] patch fired on {s['steps']} physics steps")
        print(f"[clamp] elements clamped: {100*s['clamped_fraction']:.4f}% of all (env,joint) writes")
        print(f"[clamp] |velocity| removed per physics step: {s['removed_per_step']:.4f} rad/s (summed over all DOFs)")
        print(f"[clamp] largest single-DOF excess seen: {s['max_excess']:.3f} rad/s")
    else:
        print("[clamp] clamp patch inactive or stats disabled (PhysX clamps in-solver)")
except Exception as exc:
    print(f"[clamp] no patch stats ({exc})")

env.close()
simulation_app.close()
