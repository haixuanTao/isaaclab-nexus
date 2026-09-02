#!/usr/bin/env python3
"""Per PHYSICS step, for one joint: what the actuator SAW (q, qd it used), what it
PRODUCED (tau), and the true state. Pinned gains, zero delay, airborne. The first
step where the two engines' traces diverge names the culprit."""
import argparse, sys, re
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--label", type=str, default="engine")
parser.add_argument("--joint", type=str, default="waist_yaw_joint")
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
act_of_j = None
for k, act in robot.actuators.items():
    idx = act.joint_indices.tolist() if hasattr(act.joint_indices, "tolist") else list(act.joint_indices)
    jn = [names[i] for i in idx]
    act.stiffness[:] = torch.tensor(resolve(act.cfg.stiffness, jn), device=u.device)
    act.damping[:] = torch.tensor(resolve(act.cfg.damping, jn), device=u.device)
    for b in ("positions_delay_buffer","velocities_delay_buffer","efforts_delay_buffer"):
        buf = getattr(act, b, None)
        if buf is not None: buf.set_time_lag(0); buf.reset()
    if j in idx: act_of_j = (k, act, idx.index(j))
k, act, jj = act_of_j
kp = float(act.stiffness[0, jj]); kd = float(act.damping[0, jj])
st = tt(robot.data.root_state_w).clone(); st[:, 2] += 3.0; st[:, 3] = 1.0; st[:, 4:7] = 0.0; st[:, 7:13] = 0.0
robot.write_root_state_to_sim(st)
jp = tt(robot.data.default_joint_pos).clone(); robot.write_joint_state_to_sim(position=jp, velocity=torch.zeros_like(jp))
u.scene.update(dt)
print(f"\n[at] ===== {args_cli.label} =====  joint={args_cli.joint} (actuator '{k}', kp={kp}, kd={kd})")
print(f"[at] {'pstep':>5} {'q_true':>8} {'qd_true':>8} | {'q_seen':>8} {'qd_seen':>8} {'tgt':>7} | {'tau_pd':>8} {'tau_out':>8}")
action = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
u.action_manager.process_action(action)
for p in range(10):
    q_true = float(tt(robot.data.joint_pos)[0, j]); qd_true = float(tt(robot.data.joint_vel)[0, j])
    u.action_manager.apply_action()
    u.scene.write_data_to_sim()        # actuator.compute runs inside here
    q_seen = float(act._joint_pos[0, jj]) if hasattr(act, "_joint_pos") else float("nan")
    qd_seen = float(act._joint_vel[0, jj]) if hasattr(act, "_joint_vel") else float("nan")
    tgt = float(tt(robot.data.joint_pos_target)[0, j]) if hasattr(robot.data, "joint_pos_target") else float("nan")
    tau_out = float(act.applied_effort[0, jj]) if hasattr(act, "applied_effort") else float("nan")
    tau_pd = kp * (tgt - q_seen) - kd * qd_seen
    print(f"[at] {p:5d} {q_true:+8.4f} {qd_true:+8.4f} | {q_seen:+8.4f} {qd_seen:+8.4f} {tgt:+7.3f} | {tau_pd:+8.3f} {tau_out:+8.3f}")
    u.sim.step(); u.scene.update(dt)
env.close(); simulation_app.close()
