#!/usr/bin/env python3
"""Is joint_vel readback consistent with d(joint_pos)/dt at the SAME step, or
lagged by one? A lagged velocity turns PD damping into anti-damping.
Drives one joint with a smooth sinusoidal torque; compares reported joint_vel
against finite differences of joint_pos at lags 0, 1, 2 physics steps."""
import argparse, sys, math
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
st = tt(robot.data.root_state_w).clone(); st[:, 2] += 20.0; st[:, 3] = 1.0; st[:, 4:7] = 0.0; st[:, 7:13] = 0.0
robot.write_root_state_to_sim(st)
jp = tt(robot.data.default_joint_pos).clone(); robot.write_joint_state_to_sim(position=jp, velocity=torch.zeros_like(jp))
u.scene.write_data_to_sim(); u.sim.step(); u.scene.update(u.physics_dt)
names = robot.joint_names; j = names.index("left_knee_joint"); dt = u.physics_dt
qs, vs = [], []
for i in range(80):
    tau = torch.zeros(u.num_envs, robot.num_joints, device=u.device)
    tau[:, j] = 6.0 * math.sin(2 * math.pi * 4.0 * i * dt)     # 4 Hz, +-6 Nm
    robot.set_joint_effort_target(tau)
    u.scene.write_data_to_sim(); u.sim.step(); u.scene.update(dt)
    qs.append(float(tt(robot.data.joint_pos)[0, j])); vs.append(float(tt(robot.data.joint_vel)[0, j]))
import statistics as stt
fd = [(qs[k] - qs[k-1]) / dt for k in range(1, len(qs))]          # velocity over step k-1 -> k
print(f"\n[lag] ===== {args_cli.label} =====  driving left_knee with 4 Hz torque; |q| range [{min(qs):+.3f},{max(qs):+.3f}]")
for lag in (0, 1, 2):
    errs = [abs(vs[k] - fd[k - 1 - lag]) for k in range(2 + lag, len(qs))]
    print(f"[lag]   reported joint_vel[k]  vs  (q[k-{lag}]-q[k-{lag+1}])/dt :  mean|err| = {stt.mean(errs):8.4f} rad/s")
print(f"[lag]   (typical |vel| = {stt.mean([abs(v) for v in vs]):.3f} rad/s; the lag with the SMALLEST error is the one the readback uses)")
env.close(); simulation_app.close()
