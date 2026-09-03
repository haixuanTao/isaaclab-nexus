#!/usr/bin/env python3
"""Replay a policy (with the training-time lift harness, as PPO sees it) and report
per-term observation magnitudes for the policy and critic groups."""
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--checkpoint", type=str, required=True); parser.add_argument("--envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=400); parser.add_argument("--label", type=str, default="obs")
AppLauncher.add_app_launcher_args(parser); args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, numpy as np, torch
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from agile.rl_env.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.envs); cfg.seed = 7
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; env = RslRlVecEnvWrapper(env)
policy = torch.jit.load(args_cli.checkpoint, map_location=u.device).eval()
om = u.observation_manager
obs, _ = env.reset()
stats = {}
def tt(x): return x.torch if hasattr(x, "torch") else x
with torch.inference_mode():
    for step in range(args_cli.steps):
        po = obs["policy"] if hasattr(obs, "keys") else obs
        obs, rew, dones, extras = env.step(policy(po))
        for g in om.active_terms.keys() if hasattr(om, "active_terms") else []:
            names = om._group_obs_term_names[g]; dims = om._group_obs_term_dim[g]
            t = tt(obs[g]) if hasattr(obs, "keys") and g in obs.keys() else None
            if t is None: continue
            off = 0
            for n, d in zip(names, dims):
                w = int(np.prod(d)); seg = t[:, off:off + w]; off += w
                s = stats.setdefault((g, n), {"absmax": 0.0, "p99": [], "rms": []})
                s["absmax"] = max(s["absmax"], seg.abs().max().item()); s["p99"].append(seg.abs().flatten().float().quantile(0.99).item()); s["rms"].append(seg.pow(2).mean().sqrt().item())
print(f"\n[{args_cli.label}] group/term                          absmax      p99(mean)   rms(mean)")
for (g, n), s in sorted(stats.items()):
    print(f"[{args_cli.label}] {g+'/'+n:36s} {s['absmax']:10.2f} {np.mean(s['p99']):10.3f} {np.mean(s['rms']):10.3f}")
env.close(); simulation_app.close()
