#!/usr/bin/env python3
"""Prove the Newton DC-motor envelope == PhysX DelayedDCMotor torque-speed curve.

For a sweep of joint velocities, write them into the sim, step once so the envelope
callback writes jnt_actfrcrange, read it back, and compare to isaaclab.actuators.DCMotor's
exact clip formula (the PhysX behaviour) computed independently in numpy. Reports the max
abs error between the solver-enforced [min,max] torque window and the PhysX window."""
import argparse, sys, numpy as np
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser); args_cli, hydra = parser.parse_known_args(); sys.argv=[sys.argv[0]]+hydra
app = AppLauncher(args_cli).app
import gymnasium as gym, torch, warp as wp
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_newton.physics import NewtonManager
from agile.isaaclab_extras import newton_dc_motor_envelope as env_mod
cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.envs); env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; env.reset()
robot = u.scene["robot"]; s = NewtonManager._solver; mjw = s.mjw_model; model = NewtonManager.get_model()
sat, eff, vel, _ = env_mod._build_dof_params(model)
jmap = s.mjc_jnt_to_newton_dof.numpy()  # (world, mj_jnt) -> newton dof
names = [env_mod._joint_name(l) for l in model.joint_label]
qd_start = model.joint_qd_start.numpy()
# newton dof -> joint name (first dof of each joint)
dof_name = ["?"]*int(model.joint_dof_count)
for j, st in enumerate(qd_start):
    if j < len(names): dof_name[int(st)] = names[j]
def physx_window(qd, s_, e_, v_):
    """isaaclab.actuators.DCMotor._clip_effort, exactly."""
    vlim = v_*(1.0+e_/s_); qd = np.clip(qd, -vlim, vlim)
    top = s_*(1.0-qd/v_); bot = s_*(-1.0-qd/v_)
    return np.maximum(bot, -e_), np.minimum(top, e_)
# Set the solver velocity directly and run ONLY the post-actuator callbacks (no
# integration), so the velocity the envelope kernel reads is exactly the one we compare.
qd_state = wp.to_torch(NewtonManager._state_0.joint_qd)   # flat, global newton dof
ndof_total = qd_state.numel()
vel_g = torch.tensor(vel, device=qd_state.device)         # rated vel per global dof
worst = 0.0; rows = []
for frac in [-2.0,-1.5,-1.0,-0.5,-0.1,0.0,0.1,0.5,1.0,1.5,2.0]:
    qd_state[:] = frac * vel_g[:ndof_total]               # qd/v = frac on every driven dof
    for cb in NewtonManager._post_actuator_callbacks:      # writes jnt_actfrcrange from state_0.joint_qd
        cb()
    wp.synchronize()
    qd_read = qd_state.cpu().numpy()
    afr = wp.to_torch(mjw.jnt_actfrcrange).cpu().numpy()
    if afr.ndim==2: afr = afr[None].repeat(u.num_envs,0)
    e_max=0.0; sample=None
    for w in range(afr.shape[0]):
        for mj in range(afr.shape[1]):
            d = int(jmap[w,mj]) if jmap.ndim==2 else int(jmap[mj])
            if d<0 or sat[d]<=0 or vel[d]<=0 or eff[d]<=0: continue
            qd = float(qd_read[d])
            pmin,pmax = physx_window(qd, float(sat[d]), float(eff[d]), float(vel[d]))
            nmin,nmax = float(afr[w,mj,0]), float(afr[w,mj,1])
            err = max(abs(nmin-pmin), abs(nmax-pmax))
            if err>e_max: e_max=err; sample=(dof_name[d], float(sat[d]), round(qd,1), (round(nmin,1),round(nmax,1)), (round(pmin,1),round(pmax,1)))
    worst=max(worst,e_max); rows.append((frac,e_max,sample))
print("\n[dc] frac  qd/v   max|Newton-PhysX| (Nm)   worst joint  sat  qd   Newton[min,max]   PhysX[min,max]")
for frac,e,sm in rows:
    print(f"[dc] {frac:+.1f}   {frac:+.1f}   {e:8.3f}                 {sm}")
print(f"\n[dc] WORST error over the whole sweep: {worst:.4f} Nm  ->  {'MATCH (< 0.5 Nm)' if worst<0.5 else 'MISMATCH'}", flush=True)
env.close(); app.close()
