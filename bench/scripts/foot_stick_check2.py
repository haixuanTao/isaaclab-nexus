#!/usr/bin/env python3
"""Foot adhesion, corrected:
 A) feet indexed via the CONTACT SENSOR's own body list; min/mean normal force while standing.
 B) upward force on the torso applied inside a manual physics loop (no action manager,
    so AGILE's LiftAction cannot overwrite it): do the feet leave the ground?"""
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
sb = list(sensor.body_names); rb = list(robot.body_names)
feet_s = [i for i, n in enumerate(sb) if "ankle_roll" in n]; feet_r = [i for i, n in enumerate(rb) if "ankle_roll" in n]; torso_r = rb.index("torso_link")
g = abs(u.cfg.sim.gravity[2]); W = float(tt(robot.data.default_mass)[0].sum()) * g; dt = u.physics_dt
print(f"\n[foot2] ===== {args_cli.label} =====  weight={W:.0f} N   sensor feet idx={feet_s} names={[sb[i] for i in feet_s]}   (robot idx={feet_r})")
action = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
mins, means, neg, tot = 1e9, [], 0, 0
for s in range(50):
    env.step(action)
    f = tt(sensor.data.net_forces_w)[:, feet_s, 2]
    mins = min(mins, float(f.min())); means.append(float(f.mean())); neg += int((f < -1.0).sum()); tot += f.numel()
print(f"[foot2] A) standing: foot normal force mean={sum(means)/len(means):.1f} N (weight/2 = {W/2:.0f})  min={mins:.1f} N  "
      f"samples < -1 N: {neg}/{tot} ({100*neg/tot:.1f}%)  -> {'PULLING FORCES PRESENT' if neg > 0.01*tot else 'no pulling forces'}")
# ---- B) manual loop: hold PD targets (zero action already processed), add upward force on torso ----
u.action_manager.process_action(action)
F = torch.zeros(u.num_envs, 1, 3, device=u.device); F[:, 0, 2] = args_cli.lift_factor * W; T = torch.zeros_like(F)
tid = torch.tensor([torso_r], device=u.device)
z0 = float(tt(robot.data.root_pos_w)[:, 2].mean()); fz0 = float(tt(robot.data.body_pos_w)[:, feet_r, 2].mean())
print(f"[foot2] B) +{args_cli.lift_factor:g} x weight ({args_cli.lift_factor*W:.0f} N) on torso, manual physics loop, 1 s")
print(f"[foot2] {'t(s)':>5} {'pelvis_z':>9} {'foot_z':>8} {'foot_Fz':>9}")
for s in range(200):
    u.action_manager.apply_action()
    robot.set_external_force_and_torque(F, T, body_ids=tid)   # after apply_action so nothing overwrites it
    u.scene.write_data_to_sim(); u.sim.step(); u.scene.update(dt)
    if s % 40 == 39:
        print(f"[foot2] {(s+1)*dt:5.2f} {float(tt(robot.data.root_pos_w)[:,2].mean()):9.3f} {float(tt(robot.data.body_pos_w)[:,feet_r,2].mean()):8.3f} {float(tt(sensor.data.net_forces_w)[:,feet_s,2].mean()):9.1f}")
pz = float(tt(robot.data.root_pos_w)[:, 2].mean()); fz = float(tt(robot.data.body_pos_w)[:, feet_r, 2].mean()); ff = float(tt(sensor.data.net_forces_w)[:, feet_s, 2].mean())
print(f"[foot2] B) result: pelvis {pz - z0:+.3f} m, feet {fz - fz0:+.3f} m, foot force {ff:.1f} N  -> "
      f"{'feet left the ground' if (fz - fz0) > 0.05 and ff < 20 else 'FEET STAYED ON THE GROUND under 2x weight'}")
env.close(); simulation_app.close()
