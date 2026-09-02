#!/usr/bin/env python3
"""Does the in-solver DC envelope bite once joints are moving fast?"""
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
cfg = parse_env_cfg(args_cli.task, num_envs=4); env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped
env.reset()
m = NewtonManager.get_model(); s = NewtonManager._solver; mjw = s.mjw_model; mjd = s.mjw_data
sat, eff, vel, _ = dcm._build_dof_params(m)
jm = s.mjc_jnt_to_newton_dof.numpy()
bites = 0; checked = 0; maxerr = 0.0; fastest = 0.0; examples = []
for step in range(120):
    # alternate max +/- actions every 10 steps -> big swings, high joint speeds
    a = 6.0 if (step // 10) % 2 == 0 else -6.0
    env.step(torch.full(u.action_space.shape, a, device=u.device))
    qd = NewtonManager._state_0.joint_qd.numpy()
    fr = mjw.jnt_actfrcrange.numpy()           # (nworld, njnt, 2) -- as left by the in-graph callback this step
    qfa = mjd.qfrc_actuator.numpy()             # (nworld, nv)
    for w in range(fr.shape[0]):
        for j in range(fr.shape[1]):
            d = int(jm[w, j])
            if d < 0 or sat[d] <= 0: continue
            v, e, sv, q = vel[d], eff[d], sat[d], qd[d]
            tmax = float(np.clip(sv * (1 - q / v), 0, e)); tmin = float(np.clip(sv * (-1 - q / v), -e, 0))
            got = fr[w, j]; checked += 1
            maxerr = max(maxerr, abs(got[0] - tmin), abs(got[1] - tmax))
            fastest = max(fastest, abs(q) / v)
            if abs(tmax - e) > 1e-3 or abs(tmin + e) > 1e-3:
                bites += 1
                if len(examples) < 4 and abs(q) / v > 0.9:
                    examples.append((step, w, j, round(float(q),1), round(float(v),1), [round(float(x),1) for x in got], round(float(qfa[w, d]) if d < qfa.shape[1] else float('nan'),1)))
print(f"\n[d3] range entries checked={checked}  matching DC formula (max abs err)={maxerr:.3f} Nm")
print(f"[d3] entries where the envelope was tighter than the flat effort limit: {bites} ({100*bites/max(checked,1):.1f}%)")
print(f"[d3] fastest joint reached {100*fastest:.0f}% of its velocity limit")
for ex in examples: print(f"[d3] example step={ex[0]} world={ex[1]} jnt={ex[2]} qd={ex[3]} (lim {ex[4]}) range={ex[5]} qfrc_actuator={ex[6]}")
env.close(); simulation_app.close()
