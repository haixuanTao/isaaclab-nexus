#!/usr/bin/env python3
"""Is a joint-LIMIT constraint active on waist_yaw while it sits at q~0?
A limit spring of 1e4 N.m/rad firing inside the range (inverted/degenerate
range or huge margin) is a negative stiffness -- exactly a lambda~2 mode.
Reads the GPU model's jnt_range/jnt_margin and, per PD step, the active
constraint set (efc) and the limit force on the waist DOF."""
import argparse, sys
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
def tt(x): return x.torch if hasattr(x, "torch") else x
cfg = parse_env_cfg(args_cli.task, num_envs=2); cfg.seed = 42
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; robot = u.scene["robot"]
env.reset(); names = robot.joint_names; j = names.index("waist_yaw_joint")
solver = next((o for o in gc.get_objects() if type(o).__name__ == "SolverMuJoCo"), None)
mw, mj = solver.mjw_model, solver.mj_model
def g(obj, name):
    a = getattr(obj, name, None)
    if a is None: return None
    try: return wp.to_torch(a).float().cpu().numpy() if isinstance(a, wp.array) else np.asarray(a)
    except Exception: return None
jr = g(mw, "jnt_range"); jm = g(mw, "jnt_margin"); jl = g(mw, "jnt_limited"); jsr = g(mw, "jnt_solref")
print("\n[lim] GPU jnt_range shape:", None if jr is None else jr.shape)
if jr is not None:
    r = jr.reshape(-1, 2) if jr.ndim >= 2 else jr
    bad = [(i, r[i]) for i in range(len(r)) if r[i][0] > r[i][1]]
    print(f"[lim] joints with lo > hi (INVERTED range): {len(bad)}  e.g. {bad[:4]}")
    print(f"[lim] range sample (first 8): {[tuple(np.round(x,3)) for x in r[:8]]}")
print(f"[lim] GPU jnt_margin: {None if jm is None else (jm.min(), jm.max())}   jnt_limited nonzero: {None if jl is None else int(np.count_nonzero(jl))}")
print(f"[lim] GPU jnt_solref min/max: {None if jsr is None else (jsr.min(), jsr.max())}")
print(f"[lim] CPU jnt_range lo>hi count: {int(np.sum(np.asarray(mj.jnt_range)[:,0] > np.asarray(mj.jnt_range)[:,1]))}   CPU jnt_margin max={np.asarray(mj.jnt_margin).max()}")
# which mujoco joint/dof is waist_yaw?  match by name
try:
    import mujoco
    mj_names = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(mj.njnt)]
    wy = [i for i, n in enumerate(mj_names) if n and "waist_yaw" in n]
    print(f"[lim] mujoco joint ids for waist_yaw: {wy}  names={[mj_names[i] for i in wy]}  range={[tuple(np.round(np.asarray(mj.jnt_range)[i],3)) for i in wy]}  dofadr={[int(np.asarray(mj.jnt_dofadr)[i]) for i in wy]}")
except Exception as e:
    print("[lim] name lookup failed:", e); wy = []
# PD on waist only, then per step inspect constraints
for k, act in robot.actuators.items():
    idx = act.joint_indices.tolist() if hasattr(act.joint_indices, "tolist") else list(act.joint_indices)
    keep = torch.tensor([names.index(n) == j for n in [names[i] for i in idx]], device=u.device)
    act.stiffness[:] = act.stiffness * keep; act.damping[:] = act.damping * keep
st = tt(robot.data.root_state_w).clone(); st[:, 2] += 3.0; st[:, 3] = 1.0; st[:, 4:7] = 0.0; st[:, 7:13] = 0.0
robot.write_root_state_to_sim(st); jp = tt(robot.data.default_joint_pos).clone()
robot.write_joint_state_to_sim(position=jp, velocity=torch.zeros_like(jp)); u.scene.update(u.physics_dt)
action = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device); u.action_manager.process_action(action)
d = solver.mjw_data if hasattr(solver, "mjw_data") else None
dof = int(np.asarray(mj.jnt_dofadr)[wy[0]]) if wy else -1
for p in range(8):
    u.action_manager.apply_action(); u.scene.write_data_to_sim(); u.sim.step(); u.scene.update(u.physics_dt)
    qd = float(tt(robot.data.joint_vel)[0, j]); q = float(tt(robot.data.joint_pos)[0, j])
    info = ""
    if d is not None:
        try:
            nl = g(d, "nl"); ne = g(d, "ne"); nf = g(d, "nf"); nefc = g(d, "nefc")
            qfrc_c = g(d, "qfrc_constraint"); qfrc_p = g(d, "qfrc_passive"); qfrc_a = g(d, "qfrc_applied"); qfrc_act = g(d, "qfrc_actuator")
            def at(a):
                if a is None: return float("nan")
                a = a.reshape(a.shape[0], -1) if a.ndim > 1 else a.reshape(1, -1)
                return float(a[0, dof]) if dof >= 0 and dof < a.shape[1] else float("nan")
            info = (f"nefc={None if nefc is None else int(np.ravel(nefc)[0])} nl(limits)={None if nl is None else int(np.ravel(nl)[0])} "
                    f"| waist dof: applied={at(qfrc_a):+.2f} constraint={at(qfrc_c):+.2f} passive={at(qfrc_p):+.2f} actuator={at(qfrc_act):+.2f}")
        except Exception as e: info = f"(efc read failed: {e})"
    print(f"[lim] pstep {p}: q={q:+.4f} qd={qd:+.4f}  {info}")
env.close(); simulation_app.close()
