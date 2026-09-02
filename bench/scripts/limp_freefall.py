#!/usr/bin/env python3
"""LIMP robot in free fall: all actuator gains AND effort limits zeroed, so
joint torque is exactly zero. Angular momentum must then be conserved.
Reports root orientation and total angular momentum proxy (root ang vel)."""
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
cfg = parse_env_cfg(args_cli.task, num_envs=8); cfg.seed = 42
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; robot = u.scene["robot"]
env.reset()
n = 0
for act in robot.actuators.values():
    for a in ("stiffness","damping","saturation_effort","effort_limit"):
        v = getattr(act, a, None)
        if isinstance(v, torch.Tensor): v.zero_(); n += 1
st = tt(robot.data.root_state_w).clone()
st[:, 2] += 20.0; st[:, 3] = 1.0; st[:, 4:7] = 0.0; st[:, 7:13] = 0.0
robot.write_root_state_to_sim(st)
jp = tt(robot.data.default_joint_pos).clone()
robot.write_joint_state_to_sim(position=jp, velocity=torch.zeros_like(jp))
u.scene.write_data_to_sim()
print(f"\n[lf] ===== {args_cli.label} =====  zeroed {n} actuator tensors; lifted 20 m (no contact possible)")
print(f"[lf] {'t(s)':>6} {'quat_w':>8} {'|root_ang_vel|':>15} {'|joint_vel|max':>15} {'|applied_tau|max':>17}")
dt = u.physics_dt
for i in range(120):
    u.scene.write_data_to_sim(); u.sim.step(); u.scene.update(dt)
    if i % 15 == 0 or i == 119:
        q = float(tt(robot.data.root_quat_w)[:,0].mean())
        w = float(tt(robot.data.root_ang_vel_w).norm(dim=-1).mean())
        jv = float(tt(robot.data.joint_vel).abs().max())
        tau = float(tt(robot.data.applied_torque).abs().max())
        print(f"[lf] {(i+1)*dt:6.3f} {q:8.3f} {w:15.4f} {jv:15.4f} {tau:17.4f}")
q_end = float(tt(robot.data.root_quat_w)[:,0].mean())
print(f"[lf] VERDICT: {'ORIENTATION HELD (momentum conserved)' if q_end > 0.95 else 'REORIENTED WITH ZERO TORQUE -- angular momentum NOT conserved'}")
env.close(); simulation_app.close()
