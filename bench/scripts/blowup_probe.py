#!/usr/bin/env python3
"""What goes non-finite first when the velocity clamp is off?

Bang-bang max actions on every joint (the worst thing a fresh policy does),
4 envs. Each control step: max |joint vel|, max |joint pos| beyond limit,
min contact dist, max |qfrc_actuator|, max |qfrc_constraint|; stop at the
first non-finite value and print the step before it.
"""
import argparse, sys, os
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--label", type=str, default="run"); parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser); args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, numpy as np, torch
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_newton.physics import NewtonManager
cfg = parse_env_cfg(args_cli.task, num_envs=4); env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped
robot = u.scene["robot"]; env.reset()
m = NewtonManager.get_model(); s = NewtonManager._solver; mjd = s.mjw_data; mjw = s.mjw_model
lo = m.joint_limit_lower.numpy(); hi = m.joint_limit_upper.numpy()
def tt(x): return x.torch if hasattr(x, "torch") else x
print(f"\n[bu:{args_cli.label}] env={ {k: os.environ.get(k) for k in ('AGILE_NEWTON_VEL_CLAMP','AGILE_NEWTON_DC_ENVELOPE','AGILE_NEWTON_SUBSTEPS','AGILE_NEWTON_LIMIT_KD','AGILE_NEWTON_INTEGRATOR')} }", flush=True)
opt = mjw.opt
try:
    print(f"[bu:{args_cli.label}] opt: solver={int(opt.solver)} iterations={int(opt.iterations)} ls_iterations={int(opt.ls_iterations)} cone={int(opt.cone)} timestep={float(opt.timestep):.4f}   jnt_solref[0,1:4]={mjw.jnt_solref.numpy()[0,1:4].tolist()}   jnt_solimp[0,1]={mjw.jnt_solimp.numpy()[0,1].tolist()}", flush=True)
except Exception as exc:
    print(f"[bu:{args_cli.label}] opt read failed: {exc}")
hist = []
for step in range(args_cli.steps):
    a = 6.0 if (step // 10) % 2 == 0 else -6.0
    env.step(torch.full(u.action_space.shape, a, device=u.device))
    qd = NewtonManager._state_0.joint_qd.numpy(); q = NewtonManager._state_0.joint_q.numpy()
    qfa = mjd.qfrc_actuator.numpy(); qfc = mjd.qfrc_constraint.numpy() if hasattr(mjd, "qfrc_constraint") else np.zeros(1)
    nc = int(mjd.nacon.numpy()[0]) if hasattr(mjd, "nacon") else -1
    dist = mjd.contact.dist.numpy()[:max(nc,0)] if nc > 0 else np.array([0.0])
    row = dict(step=step, max_qd=float(np.nanmax(np.abs(qd))), max_qfa=float(np.nanmax(np.abs(qfa))), max_qfc=float(np.nanmax(np.abs(qfc))),
               min_dist=float(np.nanmin(dist)), ncon=nc,
               finite=bool(np.isfinite(qd).all() and np.isfinite(qfa).all() and np.isfinite(q).all()))
    hist.append(row)
    if not row["finite"]:
        prev = hist[-2] if len(hist) > 1 else None
        bad = int(np.argmax(~np.isfinite(qd))) if not np.isfinite(qd).all() else -1
        print(f"[bu:{args_cli.label}] NON-FINITE at step {step}  first bad newton dof={bad} (joint_label={m.joint_label[np.searchsorted(m.joint_qd_start.numpy(), bad, side='right')-1].rsplit('/',1)[-1] if bad>=0 else '?'})")
        print(f"[bu:{args_cli.label}] step before: {prev}")
        # how did the worst joint evolve over the last 8 steps?
        break
    if step % 25 == 0:
        print(f"[bu:{args_cli.label}] step {step}: max|qd|={row['max_qd']:.1f} max|qfrc_act|={row['max_qfa']:.1f} max|qfrc_con|={row['max_qfc']:.1f} min_dist={row['min_dist']:+.4f} ncon={nc}", flush=True)
else:
    print(f"[bu:{args_cli.label}] SURVIVED {args_cli.steps} steps  peak max|qd|={max(h['max_qd'] for h in hist):.1f}  peak|qfrc_con|={max(h['max_qfc'] for h in hist):.1f}")
env.close(); simulation_app.close()
