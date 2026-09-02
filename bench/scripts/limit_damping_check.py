#!/usr/bin/env python3
"""Torque-free ground collapse on Newton with the joint-limit spring damping
raised from 10 to KD. If the rebound disappears and it settles near PhysX's
0.28 m, the underdamped limit springs are what hold the limp robot up."""
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--kd", type=float, default=200.0)
parser.add_argument("--ke", type=float, default=-1.0, help="-1 = leave at 1e4")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, torch, warp as wp
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_newton.physics import NewtonManager
def tt(x): return x.torch if hasattr(x, "torch") else x
cfg = parse_env_cfg(args_cli.task, num_envs=64); cfg.seed = 42
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; robot = u.scene["robot"]
env.reset()
m = NewtonManager.get_model()
kd = wp.to_torch(m.joint_limit_kd); ke = wp.to_torch(m.joint_limit_ke)
mask = kd != 0
print(f"[ld] before: limit_ke={float(ke[mask].max()):.0f} limit_kd={float(kd[mask].max()):.0f} on {int(mask.sum())} dofs")
kd[mask] = args_cli.kd
if args_cli.ke > 0: ke[mask] = args_cli.ke
flags = None
for path in ("newton", "newton._src.solvers.flags", "newton._src.solvers.solver"):
    try:
        mod = __import__(path, fromlist=["SolverNotifyFlags"]); flags = getattr(mod, "SolverNotifyFlags"); break
    except Exception: continue
if flags is not None and hasattr(NewtonManager, "_model_changes"):
    NewtonManager._model_changes.add(flags.JOINT_DOF_PROPERTIES); print("[ld] notified JOINT_DOF_PROPERTIES")
else:
    print("[ld] WARNING: could not notify solver; change may not apply")
print(f"[ld] after : limit_kd={float(wp.to_torch(m.joint_limit_kd)[mask].max()):.0f}")
for act in robot.actuators.values():
    for a in ("stiffness","damping","saturation_effort","effort_limit"):
        v = getattr(act, a, None)
        if isinstance(v, torch.Tensor): v.zero_()
dt = u.step_dt; action = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
z0 = float(tt(robot.data.root_pos_w)[:, 2].mean()); zs = []
for i in range(150):
    env.step(action)
    if i % 5 == 0 or i == 149: zs.append((round((i+1)*dt, 2), round(float(tt(robot.data.root_pos_w)[:, 2].mean()), 3)))
print(f"[ld] ===== newton limp collapse, limit_kd={args_cli.kd:g} =====  z0={z0:.3f}")
print("[ld] z(t): " + "  ".join(f"{t}s:{z}" for t, z in zs if t in (0.12, 0.42, 0.82, 1.62, 3.0) or t == zs[-1][0]))
print(f"[ld] final z = {zs[-1][1]:.3f}   (PhysX limp final: 0.279; Newton default kd=10: 0.625)")
env.close(); simulation_app.close()
