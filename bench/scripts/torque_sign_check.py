#!/usr/bin/env python3
"""For EVERY joint: apply +5 Nm from rest (gains zero, airborne) and report the
SIGN of the resulting joint velocity. A joint whose sign differs between engines
has an inverted torque axis on one of them -> PD becomes positive feedback."""
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
cfg = parse_env_cfg(args_cli.task, num_envs=2); cfg.seed = 42
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; robot = u.scene["robot"]
env.reset()
for act in robot.actuators.values():
    for a in ("stiffness","damping"):
        v = getattr(act, a, None)
        if isinstance(v, torch.Tensor): v.zero_()
    # make sure a 5 Nm command is not clipped away on weak joints
    for a in ("effort_limit","saturation_effort"):
        v = getattr(act, a, None)
        if isinstance(v, torch.Tensor): v.clamp_(min=10.0)
names = robot.joint_names; dt = u.physics_dt
def fresh():
    st = tt(robot.data.root_state_w).clone(); st[:, 2] += 20.0; st[:, 3] = 1.0; st[:, 4:7] = 0.0; st[:, 7:13] = 0.0
    robot.write_root_state_to_sim(st)
    jp = tt(robot.data.default_joint_pos).clone(); robot.write_joint_state_to_sim(position=jp, velocity=torch.zeros_like(jp))
    z = torch.zeros(u.num_envs, robot.num_joints, device=u.device)
    for _ in range(3):   # flush any delay buffers with zero commands
        robot.set_joint_effort_target(z); u.scene.write_data_to_sim(); u.sim.step(); u.scene.update(dt)
    robot.write_joint_state_to_sim(position=jp, velocity=torch.zeros_like(jp))
print(f"\n[sgn] ===== {args_cli.label} =====  +5 Nm on each joint from rest; sign of resulting qd")
out = {}
for j, n in enumerate(names):
    fresh()
    tau = torch.zeros(u.num_envs, robot.num_joints, device=u.device); tau[:, j] = 5.0
    for _ in range(3):
        robot.set_joint_effort_target(tau); u.scene.write_data_to_sim(); u.sim.step(); u.scene.update(dt)
    qd = float(tt(robot.data.joint_vel)[0, j]); out[n] = qd
    print(f"[sgn]   {n:26s} qd={qd:+8.4f}  {'+' if qd > 0 else '-' if qd < 0 else '0'}")
import json; json.dump(out, open(f"/workspace/bench/results_newton/torque_sign_{args_cli.label}.json", "w"))
env.close(); simulation_app.close()
