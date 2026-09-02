#!/usr/bin/env python3
"""Why is jnt_actfrcrange still +-1e6 with the DC envelope patch on?"""
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
env.reset()
m = NewtonManager.get_model(); s = NewtonManager._solver; mjw = s.mjw_model
sat, eff, vel, matched = dcm._build_dof_params(m)
sel = sat > 0
print(f"\n[diag] matched dofs={matched}  eff[min,max]=({eff[sel].min():.1f},{eff[sel].max():.1f})  vel[min,max]=({vel[sel].min():.3f},{vel[sel].max():.3f})")
print(f"[diag] joint_velocity_limit raw sample: {vel[sel][:6]}   joint_effort_limit sample: {eff[sel][:6]}")
fr = mjw.jnt_actfrcrange.numpy(); lim = mjw.jnt_actfrclimited.numpy()
print(f"[diag] jnt_actfrcrange shape={fr.shape} world0 lo[min]={fr[0,:,0].min():.1f} hi[max]={fr[0,:,1].max():.1f}   jnt_actfrclimited: {lim.sum()}/{lim.size} true")
print(f"[diag] actuator_forcerange present: {hasattr(mjw,'actuator_forcerange')}  "
      + (f"shape={mjw.actuator_forcerange.shape} lo[min]={mjw.actuator_forcerange.numpy()[...,0].min():.1f} hi[max]={mjw.actuator_forcerange.numpy()[...,1].max():.1f}" if hasattr(mjw,'actuator_forcerange') else ""))
print(f"[diag] actuator_forcelimited true: {mjw.actuator_forcelimited.numpy().sum() if hasattr(mjw,'actuator_forcelimited') else 'n/a'}  nu={mjw.nu if hasattr(mjw,'nu') else '?'}")
cbs = NewtonManager._post_actuator_callbacks
print(f"[diag] post-actuator callbacks registered: {len(cbs)}  names={[getattr(c,'__name__','?') for c in cbs]}   graph captured: {NewtonManager._graph is not None}")
# eager launch to prove the kernel writes
jm = s.mjc_jnt_to_newton_dof
nworld, njnt = mjw.jnt_actfrcrange.shape
dev = m.device
wp.launch(dcm._dc_envelope_kernel, dim=(nworld, njnt),
          inputs=[jm, NewtonManager._state_0.joint_qd, wp.array(sat, dtype=wp.float32, device=dev),
                  wp.array(eff, dtype=wp.float32, device=dev), wp.array(vel, dtype=wp.float32, device=dev)],
          outputs=[mjw.jnt_actfrcrange], device=dev)
fr2 = mjw.jnt_actfrcrange.numpy()
print(f"[diag] after eager launch: world0 lo[min]={fr2[0,:,0].min():.1f} hi[max]={fr2[0,:,1].max():.1f}  n_changed={(np.abs(fr2-fr)>1e-6).any(axis=-1).sum()}")
print(f"[diag] mjc_jnt_to_newton_dof world0: {jm.numpy()[0][:12]}")
env.close(); simulation_app.close()
