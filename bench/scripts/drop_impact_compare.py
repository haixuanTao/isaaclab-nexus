#!/usr/bin/env python3
"""Same drop on either engine: default pose, PD holding it, robot released from
`--height` m above its default spawn, zero actions. Per physics step: COM height,
COM acceleration (2nd difference of mass-weighted body positions), total contact
force. Reports peak force, peak COM acceleration, bounce, and settled height."""
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--label", type=str, default="drop"); parser.add_argument("--envs", type=int, default=16)
parser.add_argument("--drop", type=float, default=0.5); parser.add_argument("--steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser); args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, numpy as np, torch
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from agile.rl_env.rsl_rl.export_pruning import prepare_training_only_actions_for_evaluation
cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.envs); cfg.seed = 3; cfg.decimation = 1; cfg.sim.render_interval = 1
prepare_training_only_actions_for_evaluation(cfg)
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; env.reset()
robot = u.scene["robot"]; sensor = u.scene.sensors.get("contact_forces")
def tt(x): return x.torch if hasattr(x, "torch") else x
mass = tt(robot.data.default_mass).to(u.device); M = mass.sum(-1); dt = u.physics_dt; g = 9.81
ids = torch.arange(args_cli.envs, device=u.device)
rp = tt(robot.data.default_root_state)[ids, :7].clone(); rp[:, :3] += u.scene.env_origins[ids]; rp[:, 2] += args_cli.drop
robot.write_root_pose_to_sim(rp, env_ids=ids); robot.write_root_velocity_to_sim(torch.zeros(args_cli.envs, 6, device=u.device), env_ids=ids)
jp = tt(robot.data.default_joint_pos)[ids].clone(); robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)
u.episode_length_buf[:] = 0
def com():
    p = tt(robot.data.body_com_pos_w) if hasattr(robot.data, "body_com_pos_w") else tt(robot.data.body_pos_w)
    return (mass.unsqueeze(-1) * p).sum(1) / M.unsqueeze(-1)
hist = []; peakF = torch.zeros(args_cli.envs, device=u.device); peakA = torch.zeros(args_cli.envs, device=u.device)
zmin = torch.full((args_cli.envs,), 9.0, device=u.device); first_contact = [None] * args_cli.envs; zbounce = torch.zeros(args_cli.envs, device=u.device)
z0 = com()[:, 2].clone()
print(f"\n[{args_cli.label}] mass={float(M[0]):.1f} kg  dt={dt}  drop={args_cli.drop} m  COM z0={float(z0.mean()):.3f}", flush=True)
with torch.inference_mode():
    for step in range(args_cli.steps):
        env.step(torch.zeros(u.action_space.shape, device=u.device))
        p = com(); hist.append(p.clone()); hist = hist[-3:]
        F = tt(sensor.data.net_forces_w).sum(1).norm(dim=-1)
        peakF = torch.maximum(peakF, F)
        if len(hist) == 3:
            a = ((hist[2] - 2 * hist[1] + hist[0]) / dt**2).norm(dim=-1) / g
            peakA = torch.maximum(peakA, a)
        for e in range(args_cli.envs):
            if first_contact[e] is None and F[e] > 20: first_contact[e] = step
            if first_contact[e] is not None:
                zmin[e] = min(zmin[e], p[e, 2]); zbounce[e] = max(zbounce[e], p[e, 2] - zmin[e]) if step > first_contact[e] + 10 else zbounce[e]
        if step % 100 == 0:
            print(f"[{args_cli.label}] step {step}: COM z={float(p[:,2].mean()):.3f}  contact F mean={float(F.mean()):.0f} N", flush=True)
zf = com()[:, 2]
print(f"[{args_cli.label}] RESULT: peak contact force {float(peakF.max()):.0f} N (median over envs {float(peakF.median()):.0f})  |  peak COM accel {float(peakA.max()):.1f} g (median {float(peakA.median()):.1f})  |  bounce after landing {float(zbounce.max()):.3f} m (median {float(zbounce.median()):.3f})  |  final COM z {float(zf.mean()):.3f} vs start {float(z0.mean()):.3f}", flush=True)
env.close(); simulation_app.close()
