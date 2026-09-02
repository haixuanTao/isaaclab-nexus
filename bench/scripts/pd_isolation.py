#!/usr/bin/env python3
"""Isolate the unstable mode. Pinned nominal gains, zero delay, airborne.
  mode=only   : PD active ONLY on the traced joint, all others torque-free
  mode=except : PD active on ALL joints EXCEPT the traced joint
Trace the joint's velocity per physics step on each engine."""
import argparse, sys, re
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--label", type=str, default="engine")
parser.add_argument("--joint", type=str, default="waist_yaw_joint")
parser.add_argument("--mode", type=str, default="only", choices=["only", "except", "all"])
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
env.reset(); names = robot.joint_names; j = names.index(args_cli.joint); dt = u.physics_dt
def resolve(spec, jn, d=0.0):
    if isinstance(spec, (int, float)): return [float(spec)]*len(jn)
    out = [d]*len(jn)
    for i, n in enumerate(jn):
        for pat, val in spec.items():
            if re.fullmatch(pat, n): out[i] = float(val)
    return out
for k, act in robot.actuators.items():
    idx = act.joint_indices.tolist() if hasattr(act.joint_indices, "tolist") else list(act.joint_indices)
    jn = [names[i] for i in idx]
    kp = torch.tensor(resolve(act.cfg.stiffness, jn), device=u.device); kd = torch.tensor(resolve(act.cfg.damping, jn), device=u.device)
    keep = torch.tensor([(names.index(n) == j) if args_cli.mode == "only" else (names.index(n) != j) if args_cli.mode == "except" else True for n in jn], device=u.device)
    act.stiffness[:] = kp * keep; act.damping[:] = kd * keep
    for b in ("positions_delay_buffer", "velocities_delay_buffer", "efforts_delay_buffer"):
        buf = getattr(act, b, None)
        if buf is not None: buf.set_time_lag(0); buf.reset()
st = tt(robot.data.root_state_w).clone(); st[:, 2] += 3.0; st[:, 3] = 1.0; st[:, 4:7] = 0.0; st[:, 7:13] = 0.0
robot.write_root_state_to_sim(st)
jp = tt(robot.data.default_joint_pos).clone(); robot.write_joint_state_to_sim(position=jp, velocity=torch.zeros_like(jp)); u.scene.update(dt)
action = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device); u.action_manager.process_action(action)
qds = []; qdmax = []
for p in range(12):
    u.action_manager.apply_action(); u.scene.write_data_to_sim(); u.sim.step(); u.scene.update(dt)
    qds.append(float(tt(robot.data.joint_vel)[0, j])); qdmax.append(float(tt(robot.data.joint_vel)[0].abs().max()))
print(f"\n[iso] ===== {args_cli.label}  mode={args_cli.mode}  joint={args_cli.joint} =====")
print(f"[iso] {args_cli.joint} qd per physics step: " + " ".join(f"{v:+.4f}" for v in qds))
print(f"[iso] max|qd| over all joints per step : " + " ".join(f"{v:.3f}" for v in qdmax))
r = [abs(qds[i+1]/qds[i]) for i in range(6, 11) if abs(qds[i]) > 1e-6]
print(f"[iso] late growth ratio per step: {[round(x,2) for x in r]}   -> {'UNSTABLE' if r and min(r) > 1.3 else 'stable'}")
env.close(); simulation_app.close()
