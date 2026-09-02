#!/usr/bin/env python3
"""PD ON, joints AT target, zero velocity, airborne. Torque should be ~0.
Print applied_torque per joint on the first steps: which joints get torque, how much."""
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--label", type=str, default="engine")
parser.add_argument("--lift", type=float, default=3.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, torch
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
def tt(x): return x.torch if hasattr(x, "torch") else x
cfg = parse_env_cfg(args_cli.task, num_envs=4); cfg.seed = 42
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; robot = u.scene["robot"]
env.reset()
names = robot.joint_names
st = tt(robot.data.root_state_w).clone()
st[:, 2] += args_cli.lift; st[:, 3] = 1.0; st[:, 4:7] = 0.0; st[:, 7:13] = 0.0
robot.write_root_state_to_sim(st)
jp = tt(robot.data.default_joint_pos).clone()
robot.write_joint_state_to_sim(position=jp, velocity=torch.zeros_like(jp))
u.scene.write_data_to_sim(); u.sim.step(); u.scene.update(u.physics_dt)
print(f"\n[to] ===== {args_cli.label} =====  PD on, joints at default target, airborne")
# what the actuators think the targets are
for k, act in robot.actuators.items():
    ks = tt(getattr(act, "stiffness", None)); kd = tt(getattr(act, "damping", None))
    print(f"[to] actuator '{k}': joints={len(act.joint_indices) if hasattr(act,'joint_indices') else '?'}  "
          f"kp[min,max]=[{float(ks.min()):.1f},{float(ks.max()):.1f}]  kd[min,max]=[{float(kd.min()):.2f},{float(kd.max()):.2f}]")
action = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
for step in range(6):
    _, _, term, trunc, _ = env.step(action)
    rs = bool((term | trunc)[0]); z = float(tt(robot.data.root_pos_w)[0, 2])
    tau = tt(robot.data.applied_torque)[0]; q = tt(robot.data.joint_pos)[0]; qd = tt(robot.data.joint_vel)[0]
    tgt = tt(robot.data.joint_pos_target)[0] if hasattr(robot.data, "joint_pos_target") else None
    order = torch.argsort(tau.abs(), descending=True)[:4].tolist()
    print(f"[to] ctrl-step {step}: |tau|max={float(tau.abs().max()):8.2f}  |qd|max={float(qd.abs().max()):7.3f}  "
          f"quat_w={float(tt(robot.data.root_quat_w)[0,0]):+.3f}  z={z:6.3f}  RESET_THIS_STEP={rs}")
    for i in order:
        t_str = f" target={float(tgt[i]):+.3f}" if tgt is not None else ""
        print(f"[to]      {names[i]:26s} tau={float(tau[i]):+8.2f}  q={float(q[i]):+.3f}{t_str}  qd={float(qd[i]):+.3f}")
env.close(); simulation_app.close()
