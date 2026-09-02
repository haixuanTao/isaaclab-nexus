#!/usr/bin/env python3
"""Does the joint limit hold against the actuator? Robot held in the air (root
pose + zero root velocity rewritten every step, so no contacts), every joint
commanded to a target far past its limit through the solver-side PD. A correct
limit leaves the joint parked at its range with the motor pushing at the cap."""
import argparse, sys, os
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--label", type=str, default="lp"); parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--target", type=float, default=10.0)
AppLauncher.add_app_launcher_args(parser); args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, numpy as np, torch
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_newton.physics import NewtonManager
cfg = parse_env_cfg(args_cli.task, num_envs=8); env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; env.reset()
robot = u.scene["robot"]; names = list(robot.joint_names)
def tt(x): return x.torch if hasattr(x, "torch") else x
lim = tt(robot.data.joint_pos_limits)[0].cpu().numpy()   # (J,2)
ids = torch.arange(8, device=u.device)
pose = tt(robot.data.root_pose_w)[ids].clone(); pose[:, 2] = 2.0
tgt = torch.full((8, len(names)), float(args_cli.target), device=u.device)
tgt[4:] *= -1.0    # half the envs pushed the other way
print(f"\n[lp:{args_cli.label}] flags={ {k: os.environ.get(k) for k in ('AGILE_NEWTON_DC_ENVELOPE','AGILE_NEWTON_LIMIT_SOLREF','AGILE_NEWTON_INTEGRATOR','AGILE_NEWTON_SUBSTEPS')} }  target=+/-{args_cli.target} rad on all joints", flush=True)
mjd = NewtonManager._solver.mjw_data
worst = None
for step in range(args_cli.steps):
    robot.write_root_pose_to_sim(pose, env_ids=ids); robot.write_root_velocity_to_sim(torch.zeros(8, 6, device=u.device), env_ids=ids)
    robot.set_joint_position_target(tgt); u.scene.write_data_to_sim(); u.sim.step(); u.scene.update(u.step_dt)
    q = tt(robot.data.joint_pos); qd = tt(robot.data.joint_vel)
    over = torch.maximum(q - torch.as_tensor(lim[:, 1], device=u.device), torch.as_tensor(lim[:, 0], device=u.device) - q).clamp(min=0)
    nc = int(mjd.nacon.numpy()[0])
    if not torch.isfinite(q).all():
        print(f"[lp:{args_cli.label}] NON-FINITE at step {step}", flush=True); break
    if step % 25 == 0 or step == args_cli.steps - 1:
        j = int(over.max(dim=1).values.argmax()); k = int(over[j].argmax())
        print(f"[lp:{args_cli.label}] step {step}: max overshoot past limit={over.max().item():.3f} rad ({names[k]})  max|qd|={qd.abs().max().item():.1f}  ncon={nc}", flush=True)
env.close(); simulation_app.close()
