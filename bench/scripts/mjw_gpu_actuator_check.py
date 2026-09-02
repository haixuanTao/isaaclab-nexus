#!/usr/bin/env python3
"""Read actuator/passive parameters from the GPU mujoco-warp model (what the
solver integrates), not the CPU mirror. Also snapshot control.joint_act /
joint_target_pos / joint_f during a PD step."""
import argparse, sys, re
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, torch, numpy as np, warp as wp, gc
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_newton.physics import NewtonManager
def tt(x): return x.torch if hasattr(x, "torch") else x
cfg = parse_env_cfg(args_cli.task, num_envs=2); cfg.seed = 42
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; robot = u.scene["robot"]
env.reset()
solver = next((o for o in gc.get_objects() if type(o).__name__ == "SolverMuJoCo"), None)
mw = solver.mjw_model
def g(name):
    a = getattr(mw, name, None)
    if a is None: return None
    try: return wp.to_torch(a).float().cpu().numpy() if isinstance(a, wp.array) else np.asarray(a)
    except Exception as e: return None
print("\n[gpu] mujoco-warp (GPU) model:")
for f in ("actuator_gainprm","actuator_biasprm","actuator_gear","actuator_ctrlrange","actuator_forcerange","dof_damping","dof_armature","dof_frictionloss","jnt_stiffness","opt.timestep"):
    if f == "opt.timestep":
        ts = g_ts = None
        try: ts = wp.to_torch(mw.opt.timestep).float().cpu().numpy() if isinstance(mw.opt.timestep, wp.array) else mw.opt.timestep
        except Exception: pass
        print(f"[gpu]   opt.timestep = {ts}")
        continue
    a = g(f)
    if a is None: print(f"[gpu]   {f:20s} n/a"); continue
    nz = int(np.count_nonzero(np.abs(a) > 1e-9)); print(f"[gpu]   {f:20s} shape={a.shape} nonzero={nz} min={a.min():+.4g} max={a.max():+.4g}")
# now one PD step and look at what control carries
names = robot.joint_names; j = names.index("waist_yaw_joint")
for k, act in robot.actuators.items():
    idx = act.joint_indices.tolist() if hasattr(act.joint_indices, "tolist") else list(act.joint_indices)
    keep = torch.tensor([names.index(n) == j for n in [names[i] for i in idx]], device=u.device)
    act.stiffness[:] = act.stiffness * keep; act.damping[:] = act.damping * keep
st = tt(robot.data.root_state_w).clone(); st[:, 2] += 3.0; st[:, 3] = 1.0; st[:, 4:7] = 0.0; st[:, 7:13] = 0.0
robot.write_root_state_to_sim(st); jp = tt(robot.data.default_joint_pos).clone()
robot.write_joint_state_to_sim(position=jp, velocity=torch.zeros_like(jp)); u.scene.update(u.physics_dt)
action = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device); u.action_manager.process_action(action)
ctrl = NewtonManager.get_control()
for p in range(6):
    u.action_manager.apply_action(); u.scene.write_data_to_sim()
    row = {}
    for f in ("joint_f","joint_act","joint_target_pos","joint_target_vel"):
        a = getattr(ctrl, f, None)
        if isinstance(a, wp.array):
            t = wp.to_torch(a).float().abs(); row[f] = (float(t.max()), int((t > 1e-6).sum()))
    print(f"[gpu] pstep {p}: qd(waist_yaw)={float(tt(robot.data.joint_vel)[0,j]):+.4f}  py_tau={float(tt(robot.data.applied_torque)[0,j]):+.3f}  "
          + "  ".join(f"{f}: max={v[0]:.3f} nz={v[1]}" for f, v in row.items()))
    u.sim.step(); u.scene.update(u.physics_dt)
env.close(); simulation_app.close()
