#!/usr/bin/env python3
"""Is jnt_actfrcrange live/writable, and is any effort limit enforced in-solver?"""
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
AppLauncher.add_app_launcher_args(parser); args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, numpy as np, torch, warp as wp
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_newton.physics import NewtonManager
from agile.isaaclab_extras import newton_dc_motor_envelope as dcm
cfg = parse_env_cfg(args_cli.task, num_envs=2); env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped
robot = u.scene["robot"]; env.reset()
m = NewtonManager.get_model(); s = NewtonManager._solver; mjw = s.mjw_model; mjd = s.mjw_data
sat, eff, vel, _ = dcm._build_dof_params(m)
print(f"\n[d2] sat[4:12]={sat[4:12]}  eff[4:12]={eff[4:12]}  vel[4:12]={vel[4:12]}  joint_qd len={NewtonManager._state_0.joint_qd.shape}")
print(f"[d2] mjw.jnt_actfrcrange id={id(mjw.jnt_actfrcrange)} dtype={mjw.jnt_actfrcrange.dtype} shape={mjw.jnt_actfrcrange.shape} ptr={mjw.jnt_actfrcrange.ptr}")
print(f"[d2] mjw.jnt_actfrcrange[0,:4] before = {mjw.jnt_actfrcrange.numpy()[0,:4].tolist()}")
# 1) torch write test
t = wp.to_torch(mjw.jnt_actfrcrange); t[0, 1, 0] = -12.5; t[0, 1, 1] = 12.5; wp.synchronize()
print(f"[d2] after torch write jnt1 = {mjw.jnt_actfrcrange.numpy()[0,1].tolist()}")
# 2) eager kernel
dev = m.device; nworld, njnt = mjw.jnt_actfrcrange.shape
wp.launch(dcm._dc_envelope_kernel, dim=(nworld, njnt),
          inputs=[s.mjc_jnt_to_newton_dof, NewtonManager._state_0.joint_qd,
                  wp.array(sat, dtype=wp.float32, device=dev), wp.array(eff, dtype=wp.float32, device=dev), wp.array(vel, dtype=wp.float32, device=dev)],
          outputs=[mjw.jnt_actfrcrange], device=dev); wp.synchronize()
print(f"[d2] after kernel jnt[0:4] = {mjw.jnt_actfrcrange.numpy()[0,:4].tolist()}")
# 3) is the effort limit enforced at all? push hard PD: set targets far away, read MuJoCo actuator forces
tgt = wp.to_torch(robot._data._joint_pos_target) if hasattr(robot._data, "_joint_pos_target") else None
print(f"[d2] joint_pos_target buffer: {None if tgt is None else tuple(tgt.shape)}")
qfa = None
for step in range(40):
    act = torch.full(u.action_space.shape, 6.0, device=u.device)  # max positive action -> big PD error
    env.step(act)
    f = mjd.qfrc_actuator.numpy(); qfa = f if qfa is None else np.maximum(qfa, np.abs(f))
qmax = np.abs(qfa).max(axis=0)
print(f"[d2] max |qfrc_actuator| over 40 steps, dofs 6..34: {np.round(qmax[6:35],1).tolist()}")
print(f"[d2] effort limits (newton dofs 6..34):            {np.round(eff[6:35],1).tolist()}")
print(f"[d2] dofs where |qfrc_actuator| > eff*1.05: {int((qmax[6:35] > eff[6:35]*1.05).sum())}/29   max ratio={float((qmax[6:35]/np.maximum(eff[6:35],1e-6)).max()):.2f}")
print(f"[d2] jnt_actfrcrange[0,:4] at end = {mjw.jnt_actfrcrange.numpy()[0,:4].tolist()}")
env.close(); simulation_app.close()
