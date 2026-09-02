#!/usr/bin/env python3
"""Same single-joint PD test, but the PD lives INSIDE the solver (sim-side
joint_target_ke/kd on waist_yaw) instead of AGILE's explicit Python DCMotor.
Run with euler and implicitfast. If implicitfast stabilises it, the remedy is
'implicit actuators on Newton', and the explicit DCMotor is the incompatibility."""
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
env.reset(); names = robot.joint_names; j = names.index("waist_yaw_joint"); dt = u.physics_dt
# explicit actuator OFF everywhere
for act in robot.actuators.values():
    act.stiffness.zero_(); act.damping.zero_()
    for b in ("positions_delay_buffer","velocities_delay_buffer","efforts_delay_buffer"):
        buf = getattr(act, b, None)
        if buf is not None: buf.set_time_lag(0); buf.reset()
# solver-side PD on waist_yaw only: kp=100, kd=2.5, target 0
jid = torch.tensor([j], device=u.device)
for fn, val in (("write_joint_stiffness_to_sim_index", 100.0), ("write_joint_damping_to_sim_index", 2.5)):
    f = getattr(robot, fn, None)
    if f is None: f = getattr(robot, fn.replace('_index',''))
    v = torch.full((u.num_envs, 1), val, device=u.device)
    f(**{('stiffness' if 'stiff' in fn else 'damping'): v, 'joint_ids': jid})
st = tt(robot.data.root_state_w).clone(); st[:, 2] += 3.0; st[:, 3] = 1.0; st[:, 4:7] = 0.0; st[:, 7:13] = 0.0
robot.write_root_state_to_sim(st); jp = tt(robot.data.default_joint_pos).clone()
jp[:, j] = 0.1   # perturb 0.1 rad off target
robot.write_joint_state_to_sim(position=jp, velocity=torch.zeros_like(jp)); u.scene.update(dt)
tgt = jp.clone(); tgt[:, j] = 0.0
qds = []; qs = []
for p in range(14):
    robot.set_joint_position_target(tgt); robot.set_joint_velocity_target(torch.zeros_like(tgt))
    u.scene.write_data_to_sim(); u.sim.step(); u.scene.update(dt)
    qds.append(float(tt(robot.data.joint_vel)[0, j])); qs.append(float(tt(robot.data.joint_pos)[0, j]))
r = [abs(qds[i+1]/qds[i]) for i in range(7, 13) if abs(qds[i]) > 1e-6]
print(f"\n[sp] ===== {args_cli.label} =====  solver-side PD on waist_yaw (kp=100 kd=2.5), explicit actuator off")
print(f"[sp] waist_yaw q  per step: " + " ".join(f"{v:+.4f}" for v in qs))
print(f"[sp] waist_yaw qd per step: " + " ".join(f"{v:+.4f}" for v in qds))
print(f"[sp] verdict: {'PD INACTIVE (q stays 0.1)' if abs(qs[-1]-0.1) < 0.005 else ('UNSTABLE' if abs(qs[-1]) > 0.3 else 'PD ACTIVE & STABLE (returns toward 0)')}")
print(f"[sp] late growth ratio: {[round(x,2) for x in r]}  -> {'UNSTABLE' if r and min(r) > 1.3 else 'STABLE'}")
env.close(); simulation_app.close()
