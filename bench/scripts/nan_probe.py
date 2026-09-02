#!/usr/bin/env python3
"""Step HeightTracking-G1-v0 with zero actions and report the first NaN.

Distinguishes physics divergence (sim state itself goes NaN) from an
observation-layer bug, and names the offending quantity + env indices.

Usage (from a WBC-AGILE tree):  ./.venv/bin/python nan_probe.py --num_envs 1024
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--action_mode", type=str, default="random", choices=["zero", "random"])
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import agile.isaaclab_extras.monkey_patches  # noqa: F401,E402  applies AGILE's patches (as train.py does)
import agile.rl_env.tasks  # noqa: F401,E402  registers the gym tasks
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def as_torch(x):
    """Unwrap ProxyArray / warp array to a torch tensor, or return None."""
    if x is None:
        return None
    if hasattr(x, "torch"):
        x = x.torch
    return x if isinstance(x, torch.Tensor) else None


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]

    obs, _ = env.reset()
    act_dim = unwrapped.action_manager.total_action_dim

    def sample_action():
        if args_cli.action_mode == "zero":
            return torch.zeros(unwrapped.num_envs, act_dim, device=unwrapped.device)
        # untrained PPO policy emits roughly unit-gaussian actions
        return torch.randn(unwrapped.num_envs, act_dim, device=unwrapped.device)

    state_fields = [
        "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w",
        "joint_pos", "joint_vel", "applied_torque", "projected_gravity_b",
    ]

    def scan(step):
        """Return list of (label, n_nan, first_env) for everything currently NaN."""
        bad = []
        for name in state_fields:
            t = as_torch(getattr(robot.data, name, None))
            if t is None:
                continue
            m = ~torch.isfinite(t)
            if m.any():
                envs = m.any(dim=tuple(range(1, m.ndim))) if m.ndim > 1 else m
                bad.append((f"sim:{name}", int(m.sum()), int(envs.nonzero()[0])))
        for group in unwrapped.observation_manager.active_terms:
            for term in unwrapped.observation_manager.active_terms[group]:
                try:
                    t = as_torch(unwrapped.observation_manager.compute_group(group))
                except Exception:
                    continue
                if t is None:
                    continue
                m = ~torch.isfinite(t)
                if m.any():
                    bad.append((f"obs:{group}", int(m.sum()), int(m.any(dim=1).nonzero()[0])))
                break  # group-level check is enough to flag it
        return bad

    from isaaclab.scene import InteractiveScene
    clamp_on = getattr(InteractiveScene, "_agile_newton_vel_clamp_applied", False)
    vlim = as_torch(getattr(robot.data, "joint_vel_limits", None))
    print(f"[probe] task={args_cli.task} num_envs={unwrapped.num_envs} device={unwrapped.device}", flush=True)
    print(f"[probe] velocity clamp patch active: {clamp_on}; "
          f"configured |vel limit| range = [{vlim.abs().min():.2f}, {vlim.abs().max():.2f}]"
          if vlim is not None else f"[probe] velocity clamp patch active: {clamp_on}", flush=True)
    for step in range(args_cli.steps):
        obs, rew, term, trunc, info = env.step(sample_action())
        bad = scan(step)
        if bad:
            print(f"\n[probe] FIRST NON-FINITE at env-step {step} (sim time {step * unwrapped.step_dt:.3f}s)")
            for label, n, first_env in bad:
                print(f"          {label:28s}  non-finite elems={n:8d}  first env={first_env}")
            e = bad[0][2]
            jp = as_torch(robot.data.joint_pos)
            nan_envs = (~torch.isfinite(jp)).any(dim=1).nonzero().flatten()
            elen = unwrapped.episode_length_buf[nan_envs]
            print(f"[probe] {len(nan_envs)} of {unwrapped.num_envs} envs non-finite; "
                  f"their episode_length_buf: min={int(elen.min())} max={int(elen.max())} "
                  f"(0 == reset this step)")
            print(f"\n[probe] state of env {e}:")
            for name in state_fields:
                t = as_torch(getattr(robot.data, name, None))
                if t is not None:
                    print(f"          {name:22s} {t[e].flatten()[:8].tolist()}")
            break
        if step % 25 == 0:
            rp = as_torch(robot.data.root_pos_w)
            jv = as_torch(robot.data.joint_vel)
            print(f"[probe] step {step:4d}  finite  |root_pos|max={rp.abs().max():9.2f}  "
                  f"|joint_vel|max={jv.abs().max():9.2f}", flush=True)
    else:
        print(f"\n[probe] no NaN in {args_cli.steps} env-steps")

    env.close()
    simulation_app.close()


main()
