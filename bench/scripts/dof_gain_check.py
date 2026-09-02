#!/usr/bin/env python3
"""Torque -> acceleration gain per joint (effective inverse inertia), delays and
gains OFF. +5 Nm on one joint for 2 physics steps from rest, airborne; report
resulting |qd| for that joint. Compared across engines by name."""
import argparse, sys, json
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
cfg = parse_env_cfg(args_cli.task, num_envs=2); cfg.seed = 42
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; robot = u.scene["robot"]
env.reset(); names = robot.joint_names; dt = u.physics_dt
for act in robot.actuators.values():
    act.stiffness.zero_(); act.damping.zero_()
    for a in ("effort_limit", "saturation_effort"):
        v = getattr(act, a, None)
        if isinstance(v, torch.Tensor): v.clamp_(min=10.0)
    for b in ("positions_delay_buffer", "velocities_delay_buffer", "efforts_delay_buffer"):
        buf = getattr(act, b, None)
        if buf is not None: buf.set_time_lag(0); buf.reset()
z = torch.zeros(u.num_envs, robot.num_joints, device=u.device)
def fresh():
    st = tt(robot.data.root_state_w).clone(); st[:, 2] += 20.0; st[:, 3] = 1.0; st[:, 4:7] = 0.0; st[:, 7:13] = 0.0
    robot.write_root_state_to_sim(st)
    jp = tt(robot.data.default_joint_pos).clone(); robot.write_joint_state_to_sim(position=jp, velocity=torch.zeros_like(jp))
    robot.set_joint_effort_target(z); u.scene.write_data_to_sim(); u.sim.step(); u.scene.update(dt)
    robot.write_joint_state_to_sim(position=jp, velocity=torch.zeros_like(jp)); u.scene.update(dt)
out = {}
print(f"\n[dg] ===== {args_cli.label} =====  +5 Nm x 2 physics steps, gains/delays off")
for j, n in enumerate(names):
    fresh(); tau = z.clone(); tau[:, j] = 5.0
    for _ in range(2):
        robot.set_joint_effort_target(tau); u.scene.write_data_to_sim(); u.sim.step(); u.scene.update(dt)
    qd = float(tt(robot.data.joint_vel)[0, j]); out[n] = qd
json.dump(out, open(f"/workspace/bench/results_newton/dof_gain_{args_cli.label}.json", "w"))
print("[dg] done"); env.close(); simulation_app.close()
