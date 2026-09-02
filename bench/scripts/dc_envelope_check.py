#!/usr/bin/env python3
"""Is the get-up powered by torque a real motor could not deliver?

Switching to implicit (solver-side PD) actuators for Newton dropped AGILE's
DelayedDCMotor saturation curve. A DC motor's available torque falls with speed:

    tau_max(qd) = clip(sat * (1 - qd/vel),  0,  eff)
    tau_min(qd) = clip(sat * (-1 - qd/vel), -eff, 0)

The implicit drive has no such limit -- only a flat effort clip. This replays the
trained policy and reports how much of the applied torque lies outside the DC
envelope at the speed the joint is actually moving.
"""
import argparse, sys, re
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--keep-assist", action="store_true", help="keep the training-only lift/harness actions (default: removed, as in eval.py)")
parser.add_argument("--envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=750)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app

import gymnasium as gym, numpy as np, torch
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from agile.rl_env.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

# DC-motor envelope from the explicit config (AGILE_NEWTON_IMPLICIT_ACTUATORS=0)
GROUPS = {
    "legs":  dict(sat=180.0, pat=[r".*_hip_.*_joint", r".*_knee_joint"]),
    "feet":  dict(sat=80.0,  pat=[r".*_ankle_.*_joint"]),
    "waist": dict(sat=120.0, pat=[r"waist_.*_joint"]),
    "arms":  dict(sat=130.0, pat=[r".*_shoulder_.*_joint", r".*_elbow_joint", r".*_wrist_.*_joint"]),
}

cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.envs)
if not args_cli.keep_assist:
    from agile.rl_env.rsl_rl.export_pruning import prepare_training_only_actions_for_evaluation
    _removed, _held = prepare_training_only_actions_for_evaluation(cfg)
    print(f"[assist] removed training-only actions {_removed}; held at default {_held}", flush=True)

cfg.seed = 42
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped
robot = u.scene["robot"]
env = RslRlVecEnvWrapper(env)
policy = torch.jit.load(args_cli.checkpoint, map_location=u.device).eval()

def tt(x): return x.torch if hasattr(x, "torch") else x

names = list(robot.joint_names)
sat = np.zeros(len(names)); 
for j, n in enumerate(names):
    for g in GROUPS.values():
        if any(re.fullmatch(p, n) for p in g["pat"]):
            sat[j] = g["sat"]; break
eff = tt(robot.data.joint_effort_limits)[0].detach().cpu().numpy().astype(float)
vel = tt(robot.data.joint_velocity_limits)[0].detach().cpu().numpy().astype(float)
kp  = tt(robot.data.joint_stiffness)[0].detach().cpu().numpy().astype(float)
kd  = tt(robot.data.joint_damping)[0].detach().cpu().numpy().astype(float)
print(f"\n[dc] joints={len(names)}  sat set for {(sat>0).sum()}  kp[min,max]=({kp.min():.0f},{kp.max():.0f})  "
      f"eff[min,max]=({eff.min():.0f},{eff.max():.0f})  vel[min,max]=({vel.min():.0f},{vel.max():.0f})", flush=True)

obs, _ = env.reset()
viol = np.zeros(len(names)); tot = 0
excess_sum = np.zeros(len(names)); peak_vel = np.zeros(len(names)); peak_tau = np.zeros(len(names))
worst = []
with torch.inference_mode():
    for step in range(args_cli.steps):
        po = obs["policy"] if (hasattr(obs, "keys") and "policy" in obs.keys()) else obs
        act = policy(po)
        obs, _, _, _ = env.step(act)
        q   = tt(robot.data.joint_pos).detach().cpu().numpy()
        qd  = tt(robot.data.joint_vel).detach().cpu().numpy()
        tgt = tt(robot.data.joint_pos_target).detach().cpu().numpy()
        tau = np.clip(kp * (tgt - q) - kd * qd, -eff, eff)      # solver-side PD drive
        tmax = np.clip(sat * (1.0 - qd / np.maximum(vel, 1e-6)),  0.0, eff)
        tmin = np.clip(sat * (-1.0 - qd / np.maximum(vel, 1e-6)), -eff, 0.0)
        over = np.maximum(tau - tmax, 0.0) + np.maximum(tmin - tau, 0.0)
        viol += (over > 1e-6).sum(axis=0); tot += tau.shape[0]
        excess_sum += over.sum(axis=0)
        peak_vel = np.maximum(peak_vel, np.abs(qd).max(axis=0))
        peak_tau = np.maximum(peak_tau, np.abs(tau).max(axis=0))
        if step % 50 == 0:
            rng_msg = ""
            try:
                from isaaclab_newton.physics import NewtonManager
                fr = NewtonManager._solver.mjw_model.jnt_actfrcrange.numpy()   # (nworld, njnt, 2)
                lo, hi = fr[0, :, 0], fr[0, :, 1]
                rng_msg = f"  jnt_actfrcrange world0: min_lo={lo.min():.1f} max_hi={hi.max():.1f} n_asym={(np.abs(lo+hi)>1e-3).sum()}"
            except Exception as exc:
                rng_msg = f"  (range read failed: {exc})"
            print(f"[dc] step {step}/{args_cli.steps}  outside-envelope now: "
                  f"{100*(over>1e-6).mean():.1f}%{rng_msg}", flush=True)

frac = 100 * viol / max(tot, 1)
print(f"\n[dc] samples per joint: {tot}")
print(f"[dc] OVERALL: {frac.mean():.1f}% of (joint,step) samples demand torque the DC motor could not deliver")
order = np.argsort(-frac)
print("[dc] worst joints:")
for j in order[:10]:
    print(f"[dc]   {names[j]:<28} outside {frac[j]:5.1f}%   peak|tau|={peak_tau[j]:6.1f} (eff {eff[j]:5.1f}, sat {sat[j]:5.0f})"
          f"   peak|qd|={peak_vel[j]:6.1f} (vel lim {vel[j]:5.1f})  mean excess={excess_sum[j]/max(tot,1):5.1f} Nm")
print(f"[dc] joints exceeding their velocity limit at peak: "
      f"{int((peak_vel > vel).sum())}/{len(names)}")
env.close(); simulation_app.close()
