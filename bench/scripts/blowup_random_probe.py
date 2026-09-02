#!/usr/bin/env python3
"""Training-like random actions (what PPO does at iteration 0-5), many envs:
which constraint type explodes first -- joint limits or contacts?"""
import argparse, sys, os
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--label", type=str, default="rnd"); parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--envs", type=int, default=256); parser.add_argument("--std", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser); args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, numpy as np, torch
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_newton.physics import NewtonManager
cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.envs); env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped
env.reset(); torch.manual_seed(0)
m = NewtonManager.get_model(); s = NewtonManager._solver; mjd = s.mjw_data; mjw = s.mjw_model
try:
    import mujoco_warp as mjwp
    TYPE = {int(getattr(mjwp.ConstraintType, k)): k for k in dir(mjwp.ConstraintType) if k.isupper()}
except Exception:
    TYPE = {}
print(f"\n[rnd:{args_cli.label}] envs={args_cli.envs} std={args_cli.std}  flags={ {k: os.environ.get(k) for k in ('AGILE_NEWTON_VEL_CLAMP','AGILE_NEWTON_DC_ENVELOPE','AGILE_NEWTON_LIMIT_SOLREF','AGILE_NEWTON_SUBSTEPS')} }", flush=True)
peak = {}
for step in range(args_cli.steps):
    a = torch.randn(u.action_space.shape, device=u.device) * args_cli.std
    env.step(a)
    qd = NewtonManager._state_0.joint_qd.numpy()
    nefc = int(mjd.nefc.numpy().max()) if hasattr(mjd, "nefc") else 0
    ef = mjd.efc.force.numpy(); et = mjd.efc.type.numpy()
    mask = np.isfinite(ef)
    by = {}
    for t_id in np.unique(et[mask]):
        sel = (et == t_id) & mask
        by[TYPE.get(int(t_id), str(int(t_id)))] = float(np.abs(ef[sel]).max()) if sel.any() else 0.0
    for k, v in by.items(): peak[k] = max(peak.get(k, 0.0), v)
    finite = bool(np.isfinite(qd).all() and mask.all())
    if not finite or step % 100 == 0:
        nc = int(mjd.nacon.numpy()[0]); dist = mjd.contact.dist.numpy()[:max(nc, 0)]
        print(f"[rnd:{args_cli.label}] step {step}: max|qd|={np.nanmax(np.abs(qd)):.1f} ncon={nc} min_dist={np.nanmin(dist) if nc else 0:+.4f}  efc by type: " +
              ", ".join(f"{k}={v:.0f}" for k, v in sorted(by.items())), flush=True)
    if not finite:
        bad = ~np.isfinite(ef); print(f"[rnd:{args_cli.label}] NON-FINITE at step {step}: non-finite efc rows by type: " +
              ", ".join(f"{TYPE.get(int(t),str(int(t)))}={int(((et==t)&bad).sum())}" for t in np.unique(et[bad])) +
              f"   non-finite joint_qd: {int((~np.isfinite(qd)).sum())}", flush=True)
        break
else:
    print(f"[rnd:{args_cli.label}] SURVIVED {args_cli.steps} steps", flush=True)
print(f"[rnd:{args_cli.label}] peak |efc.force| by type: " + ", ".join(f"{k}={v:.0f}" for k, v in sorted(peak.items())))
env.close(); simulation_app.close()
