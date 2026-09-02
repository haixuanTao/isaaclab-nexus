#!/usr/bin/env python3
"""Are the feet stuck to the ground?
 A) standing: min foot normal force over envs/time -- negative == adhesion.
 B) lift test: apply an upward external force on the torso of 2x body weight
    for 1 s. Feet must leave the ground (foot contact force -> 0, foot z rises).
    If the pelvis cannot rise and the feet stay planted, contact is holding them."""
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--label", type=str, default="engine")
parser.add_argument("--lift_factor", type=float, default=2.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, torch
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
def tt(x): return x.torch if hasattr(x, "torch") else x
cfg = parse_env_cfg(args_cli.task, num_envs=16); cfg.seed = 42
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; robot = u.scene["robot"]; sensor = u.scene.sensors["contact_forces"]
env.reset()
names = robot.body_names
feet = [i for i, n in enumerate(names) if "ankle_roll" in n]
torso = names.index("torso_link")
g = abs(u.cfg.sim.gravity[2]); mass = float(tt(robot.data.default_mass)[0].sum()); W = mass * g
action = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
dt = u.step_dt
print(f"\n[foot] ===== {args_cli.label} =====  mass={mass:.2f} kg  weight={W:.0f} N  feet bodies={[names[i] for i in feet]}")
# ---- A) settle and watch foot normal forces ----
minfz = 1e9; negcount = 0; total = 0
for s in range(50):
    env.step(action)
    f = tt(sensor.data.net_forces_w)[:, feet, 2]   # (E, 2)
    minfz = min(minfz, float(f.min())); negcount += int((f < -1.0).sum()); total += f.numel()
fz = tt(sensor.data.net_forces_w)[:, feet, 2]
print(f"[foot] A) standing 1 s: foot normal force mean={float(fz.mean()):.1f} N  min over run={minfz:.1f} N  "
      f"samples < -1 N: {negcount}/{total}  -> {'ADHESION (pulling forces)' if negcount > total*0.01 else 'no pulling forces'}")
z_ped0 = float(tt(robot.data.root_pos_w)[:, 2].mean()); fz0 = float(tt(robot.data.body_pos_w)[:, feet, 2].mean())
# ---- B) lift the torso with 2x weight ----
F = torch.zeros(u.num_envs, 1, 3, device=u.device); F[:, 0, 2] = args_cli.lift_factor * W
T = torch.zeros_like(F)
print(f"[foot] B) applying +{args_cli.lift_factor:g} x weight = {args_cli.lift_factor*W:.0f} N upward on torso for 1 s")
print(f"[foot] {'t(s)':>5} {'pelvis_z':>9} {'foot_z':>8} {'foot_dz':>8} {'foot_Fz':>8}")
for s in range(50):
    robot.set_external_force_and_torque(F, T, body_ids=torch.tensor([torso], device=u.device))
    env.step(action)
    if s % 10 == 9:
        pz = float(tt(robot.data.root_pos_w)[:, 2].mean()); fzz = float(tt(robot.data.body_pos_w)[:, feet, 2].mean())
        ff = float(tt(sensor.data.net_forces_w)[:, feet, 2].mean())
        print(f"[foot] {(s+1)*dt:5.2f} {pz:9.3f} {fzz:8.3f} {fzz-fz0:+8.3f} {ff:8.1f}")
pz = float(tt(robot.data.root_pos_w)[:, 2].mean()); fzz = float(tt(robot.data.body_pos_w)[:, feet, 2].mean()); ff = float(tt(sensor.data.net_forces_w)[:, feet, 2].mean())
lifted = (fzz - fz0) > 0.05 and ff < 20.0
print(f"[foot] B) result: pelvis rose {pz - z_ped0:+.3f} m, feet rose {fzz - fz0:+.3f} m, foot contact force {ff:.1f} N  "
      f"-> {'FEET LEFT THE GROUND (not stuck)' if lifted else 'FEET DID NOT LEAVE THE GROUND under 2x weight  <<< STUCK'}")
env.close(); simulation_app.close()
