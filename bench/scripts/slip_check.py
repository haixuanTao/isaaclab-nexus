#!/usr/bin/env python3
"""Effective foot-ground friction: ramp a horizontal force on the torso of a
standing robot (PD holding pose) and record the tangential/normal foot force
ratio at the moment the feet start sliding. That ratio IS the friction coefficient
the solver is using."""
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
cfg = parse_env_cfg(args_cli.task, num_envs=32); cfg.seed = 42
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; robot = u.scene["robot"]; sensor = u.scene.sensors["contact_forces"]
env.reset()
sb = list(sensor.body_names); rb = list(robot.body_names)
fs = [i for i, n in enumerate(sb) if "ankle_roll" in n]; fr = [i for i, n in enumerate(rb) if "ankle_roll" in n]; torso = rb.index("torso_link")
action = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
for _ in range(25): env.step(action)          # settle
u.action_manager.process_action(action); dt = u.physics_dt
W = float(tt(robot.data.default_mass)[0].sum()) * abs(u.cfg.sim.gravity[2])
tid = torch.tensor([torso], device=u.device)
x0 = tt(robot.data.body_pos_w)[:, fr, 0].clone()
slip_ratio = torch.full((u.num_envs,), float("nan"), device=u.device)
print(f"\n[slip] ===== {args_cli.label} =====  ramping +x force on torso 0 -> 1.2 W over 1.2 s; slip = foot moved > 2 cm")
for s in range(240):
    Fx = W * (s / 200.0)
    F = torch.zeros(u.num_envs, 1, 3, device=u.device); F[:, 0, 0] = Fx
    u.action_manager.apply_action(); robot.set_external_force_and_torque(F, torch.zeros_like(F), body_ids=tid)
    u.scene.write_data_to_sim(); u.sim.step(); u.scene.update(dt)
    f = tt(sensor.data.net_forces_w)[:, fs, :]            # (E,2,3)
    ft = f[..., 0].sum(1); fn = f[..., 2].sum(1).clamp(min=1.0)
    moved = (tt(robot.data.body_pos_w)[:, fr, 0] - x0).abs().max(1).values > 0.02
    new = moved & torch.isnan(slip_ratio)
    slip_ratio[new] = (ft / fn)[new].abs()
    if s % 60 == 59: print(f"[slip] t={(s+1)*dt:4.2f}s Fx={Fx:5.0f}N  slipped envs={int(moved.sum())}/{u.num_envs}  mean|Ft/Fn|={float((ft/fn).abs().mean()):.3f}")
r = slip_ratio[~torch.isnan(slip_ratio)]
print(f"[slip] envs that slipped: {r.numel()}/{u.num_envs}   Ft/Fn at slip: median={float(r.median()) if r.numel() else float('nan'):.3f}  "
      f"p10={float(r.quantile(0.1)) if r.numel() else float('nan'):.3f}  p90={float(r.quantile(0.9)) if r.numel() else float('nan'):.3f}")
print(f"[slip] (PhysX 'multiply' with terrain 1.0 and randomised foot mu 0.2-1.5 -> expect a SPREAD; MuJoCo 'max' -> expect ~1.0 for everyone)")
env.close(); simulation_app.close()
