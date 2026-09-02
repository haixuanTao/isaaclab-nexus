#!/usr/bin/env python3
"""Torso inertia BEFORE any reset/randomization vs AFTER env.reset(), vs the
authored USD value. Separates importer discrepancy from mass-randomization draws."""
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--label", type=str, default="engine")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, torch
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
def tt(x): return x.torch if hasattr(x, "torch") else x
AUTH = {"torso_link": (0.12179522, 0.10977251, 0.027373321), "pelvis": (0.0105548855, 0.009314782, 0.007918497),
        "left_knee_link": (0.011277774, 0.011380441, 0.0014645847)}
cfg = parse_env_cfg(args_cli.task, num_envs=4); cfg.seed = 42
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; robot = u.scene["robot"]
names = list(robot.body_names)
def report(tag):
    print(f"[ip] --- {tag} ---")
    for f in ("default_inertia", "body_inertia"):
        I = tt(getattr(robot.data, f, None)); m = tt(robot.data.default_mass)
        if I is None: print(f"[ip]   {f}: n/a"); continue
        for n, auth in AUTH.items():
            i = names.index(n); d = I[0, i].reshape(3, 3); tr = float(d.trace()); ta = sum(auth)
            print(f"[ip]   {f:16s} {n:14s} mass={float(m[0,i]):.3f}  trace={tr:.5f}  authored={ta:.5f}  ratio={tr/ta:.3f}  "
                  f"diag={[round(float(d[k,k]),5) for k in range(3)]}  envs-trace-spread=[{float(I[:,i].reshape(-1,3,3).diagonal(dim1=1,dim2=2).sum(1).min()):.4f},{float(I[:,i].reshape(-1,3,3).diagonal(dim1=1,dim2=2).sum(1).max()):.4f}]")
print(f"\n[ip] ===== {args_cli.label} =====")
report("BEFORE env.reset() (fresh model)")
env.reset()
report("AFTER env.reset() (mass/com randomization events applied)")
env.close(); simulation_app.close()
