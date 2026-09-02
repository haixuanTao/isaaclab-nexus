#!/usr/bin/env python3
"""Is the negative foot 'normal force' a real pull or a sign artifact?
Per physics step, one env: foot Fz from the contact sensor, foot z, foot vz.
If Fz<0 events are followed by DOWNWARD foot acceleration, the contact is
genuinely pulling. If the foot moves UP at those instants, it's a sign artifact."""
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); parser.add_argument("--task", type=str, default="HeightTracking-G1-v0"); parser.add_argument("--label", type=str, default="engine")
AppLauncher.add_app_launcher_args(parser); args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
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
fs = [i for i, n in enumerate(sb) if "ankle_roll" in n]; fr = [i for i, n in enumerate(rb) if "ankle_roll" in n]
action = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device); u.action_manager.process_action(action); dt = u.physics_dt
prev_vz = None; neg_events = []; pos_events = []
Fz_hist = []
for s in range(400):   # 2 s of physics at 5 ms
    u.action_manager.apply_action(); u.scene.write_data_to_sim(); u.sim.step(); u.scene.update(dt)
    Fz = tt(sensor.data.net_forces_w)[:, fs, 2]            # (E,2)
    vz = tt(robot.data.body_lin_vel_w)[:, fr, 2]           # (E,2)
    if prev_vz is not None:
        az = (vz - prev_vz) / dt                            # accel over the step in which Fz acted
        m = Fz < -50.0
        if m.any(): neg_events.append(az[m].mean().item())
        p = Fz > 50.0
        if p.any(): pos_events.append(az[p].mean().item())
    prev_vz = vz.clone(); Fz_hist.append(Fz.clone())
Fz_all = torch.stack(Fz_hist)
print(f"\n[pull] ===== {args_cli.label} =====  400 physics steps, 16 envs, 2 feet")
print(f"[pull] Fz < -50 N events: {int((Fz_all < -50).sum())}   Fz > 50 N events: {int((Fz_all > 50).sum())}   min Fz={float(Fz_all.min()):.0f}  max Fz={float(Fz_all.max()):.0f}")
if neg_events: print(f"[pull] mean foot z-accel during NEGATIVE-force steps: {sum(neg_events)/len(neg_events):+.1f} m/s^2   ({len(neg_events)} steps)")
if pos_events: print(f"[pull] mean foot z-accel during POSITIVE-force steps: {sum(pos_events)/len(pos_events):+.1f} m/s^2   ({len(pos_events)} steps)")
if neg_events:
    v = sum(neg_events)/len(neg_events)
    print(f"[pull] VERDICT: {'GENUINE PULL (foot accelerates DOWN when force is negative)' if v < -2 else 'sign artifact or no net pull (foot not accelerating down)'}")
env.close(); simulation_app.close()
