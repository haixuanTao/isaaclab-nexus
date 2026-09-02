#!/usr/bin/env python3
"""torque_origin with ALL per-env randomization neutralised: identical nominal
kp/kd on every joint, zero actuator delay, on both engines. If the engines now
agree, the earlier divergence was a random-draw artifact, not physics."""
import argparse, sys, re
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--label", type=str, default="engine")
parser.add_argument("--lift", type=float, default=3.0)
parser.add_argument("--steps", type=int, default=25)
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

def resolve(spec, jnames, default=0.0):
    """cfg stiffness/damping may be a float or {regex: value}; resolve per joint."""
    if isinstance(spec, (int, float)): return [float(spec)] * len(jnames)
    out = [default] * len(jnames)
    for i, n in enumerate(jnames):
        for pat, val in spec.items():
            if re.fullmatch(pat, n): out[i] = float(val)
    return out

# ---- pin gains to NOMINAL cfg values, kill delays ----
for k, act in robot.actuators.items():
    jn = [names[i] for i in act.joint_indices.tolist()] if hasattr(act.joint_indices, "tolist") else [names[i] for i in act.joint_indices]
    kp = torch.tensor(resolve(act.cfg.stiffness, jn), device=u.device); kd = torch.tensor(resolve(act.cfg.damping, jn), device=u.device)
    act.stiffness[:] = kp; act.damping[:] = kd
    for b in ("positions_delay_buffer", "velocities_delay_buffer", "efforts_delay_buffer"):
        buf = getattr(act, b, None)
        if buf is not None: buf.set_time_lag(0); buf.reset()
    print(f"[tf] '{k}': kp={[round(v,1) for v in kp.tolist()[:4]]}.. kd={[round(v,2) for v in kd.tolist()[:4]]}..  delay=0")

st = tt(robot.data.root_state_w).clone(); st[:, 2] += args_cli.lift; st[:, 3] = 1.0; st[:, 4:7] = 0.0; st[:, 7:13] = 0.0
robot.write_root_state_to_sim(st)
jp = tt(robot.data.default_joint_pos).clone(); robot.write_joint_state_to_sim(position=jp, velocity=torch.zeros_like(jp))
u.scene.write_data_to_sim(); u.sim.step(); u.scene.update(u.physics_dt)
print(f"\n[tf] ===== {args_cli.label} =====  nominal gains, zero delay, PD on, airborne at {args_cli.lift} m")
action = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
for step in range(args_cli.steps):
    _, _, term, trunc, _ = env.step(action)
    tau = tt(robot.data.applied_torque)[0]; qd = tt(robot.data.joint_vel)[0]
    if step % 4 == 0 or step == args_cli.steps - 1:
        print(f"[tf] ctrl-step {step:2d}: |tau|max={float(tau.abs().max()):8.2f}  |qd|max={float(qd.abs().max()):7.3f}  "
              f"quat_w={float(tt(robot.data.root_quat_w)[0,0]):+.3f}  z={float(tt(robot.data.root_pos_w)[0,2]):6.3f}  reset={bool((term|trunc)[0])}")
env.close(); simulation_app.close()
