#!/usr/bin/env python3
"""Is control.joint_f (the torque the solver integrates) what Isaac Lab's Python
actuator wrote, or is it overwritten inside Newton's step by the in-graph
adapter? Compare joint_f right after write_data_to_sim vs right after sim.step,
on waist_yaw, single-joint PD, pinned gains."""
import argparse, sys, re
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--joint", type=str, default="waist_yaw_joint")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, torch, warp as wp
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_newton.physics import NewtonManager
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
    keep = torch.tensor([names.index(n) == j for n in jn], device=u.device)
    act.stiffness[:] = torch.tensor(resolve(act.cfg.stiffness, jn), device=u.device) * keep
    act.damping[:] = torch.tensor(resolve(act.cfg.damping, jn), device=u.device) * keep
    for b in ("positions_delay_buffer","velocities_delay_buffer","efforts_delay_buffer"):
        buf = getattr(act, b, None)
        if buf is not None: buf.set_time_lag(0); buf.reset()
ad = getattr(NewtonManager, "_adapter", None)
print(f"\n[jf] NewtonManager._adapter = {type(ad).__name__ if ad else None};  all_graphable = {NewtonManager._is_all_graphable() if hasattr(NewtonManager,'_is_all_graphable') else '?'}")
try:
    acts = getattr(ad, "_actuators", None) or getattr(ad, "actuators", None)
    print(f"[jf] adapter-registered actuators: {len(acts) if acts is not None else 'n/a'}  types={[type(a).__name__ for a in acts][:6] if acts else ''}")
except Exception as e: print("[jf] adapter introspection failed:", e)
st = tt(robot.data.root_state_w).clone(); st[:, 2] += 3.0; st[:, 3] = 1.0; st[:, 4:7] = 0.0; st[:, 7:13] = 0.0
robot.write_root_state_to_sim(st)
jp = tt(robot.data.default_joint_pos).clone(); robot.write_joint_state_to_sim(position=jp, velocity=torch.zeros_like(jp)); u.scene.update(dt)
action = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device); u.action_manager.process_action(action)
# map isaaclab joint j -> newton dof index for env 0 (find via robot's binding if exposed; else search by writing a marker)
ctrl = NewtonManager.get_control()
def jf_all(): return wp.to_torch(ctrl.joint_f).float().clone()
print(f"[jf] control.joint_f shape={tuple(jf_all().shape)}")
print(f"[jf] {'pstep':>5} {'qd_true':>9} | {'py_tau(act)':>11} | {'joint_f AFTER python write':>26} | {'joint_f AFTER sim.step':>22}")
for p in range(8):
    qd_true = float(tt(robot.data.joint_vel)[0, j])
    u.action_manager.apply_action(); u.scene.write_data_to_sim()
    py_tau = float(tt(robot.data.applied_torque)[0, j])
    f_before = jf_all()
    nz_b = (f_before.abs() > 1e-6).nonzero().flatten().tolist()
    u.sim.step()
    f_after = jf_all()
    nz_a = (f_after.abs() > 1e-6).nonzero().flatten().tolist()
    vb = [round(float(f_before[i]),3) for i in nz_b[:3]]; va = [round(float(f_after[i]),3) for i in nz_a[:3]]
    print(f"[jf] {p:5d} {qd_true:+9.4f} | {py_tau:+11.3f} | nz@{nz_b[:3]}={vb:<26} | nz@{nz_a[:3]}={va}")
    u.scene.update(dt)
env.close(); simulation_app.close()
