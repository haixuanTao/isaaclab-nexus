#!/usr/bin/env python3
"""Which COM acceleration is real? Compare, per physics step and env:
  (pos)   2nd difference of the mass-weighted body positions (Isaac body_pos_w)   -- unambiguous
  (vel)   1st difference of Isaac's body_com_lin_vel_w-based COM velocity
  (mjcom) 1st difference of MuJoCo subtree_linvel of the root body (its own COM velocity)
  (qacc)  MuJoCo qacc on the root translational dofs
against the root-dof constraint force and the contact sensor."""
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--checkpoint", type=str, required=True); parser.add_argument("--envs", type=int, default=16); parser.add_argument("--steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser); args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, numpy as np, torch
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_newton.physics import NewtonManager
from agile.rl_env.rsl_rl.export_pruning import prepare_training_only_actions_for_evaluation
cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.envs); cfg.seed = 7; cfg.decimation = 1; cfg.sim.render_interval = 1
prepare_training_only_actions_for_evaluation(cfg)
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped
from agile.rl_env.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
wenv = RslRlVecEnvWrapper(env); obs, _ = wenv.reset()
policy = torch.jit.load(args_cli.checkpoint, map_location=u.device).eval()
robot = u.scene["robot"]; mjd = NewtonManager._solver.mjw_data; sensor = u.scene.sensors.get("contact_forces")
def tt(x): return x.torch if hasattr(x, "torch") else x
mass = tt(robot.data.default_mass).to(u.device); M = mass.sum(-1); dt = u.physics_dt; g = 9.81
print(f"\n[cr] total mass per robot {float(M[0]):.1f} kg, dt={dt}, mjd has subtree_linvel: {hasattr(mjd,'subtree_linvel')}, qacc: {hasattr(mjd,'qacc')}", flush=True)
def com_pos():
    p = tt(robot.data.body_com_pos_w) if hasattr(robot.data, "body_com_pos_w") else tt(robot.data.body_pos_w)
    return (mass.unsqueeze(-1) * p).sum(1) / M.unsqueeze(-1)
def com_vel_isaac():
    v = tt(robot.data.body_com_lin_vel_w) if hasattr(robot.data, "body_com_lin_vel_w") else tt(robot.data.body_lin_vel_w)
    return (mass.unsqueeze(-1) * v).sum(1) / M.unsqueeze(-1)
def com_vel_mj():
    if not hasattr(mjd, "subtree_linvel"): return None
    sv = torch.as_tensor(mjd.subtree_linvel.numpy(), device=u.device)   # (nworld, nbody, 3); body 1 = robot root (0 = world)
    return sv[:, 1]
rows = []; p_hist = []; vi_prev = None; vm_prev = None
with torch.inference_mode():
    for step in range(args_cli.steps):
        po = obs["policy"] if hasattr(obs, "keys") else obs
        obs, rew, dones, extras = wenv.step(policy(po))
        p = com_pos(); vi = com_vel_isaac(); vm = com_vel_mj()
        p_hist.append(p.clone()); p_hist = p_hist[-3:]
        qc = torch.as_tensor(mjd.qfrc_constraint.numpy(), device=u.device)[:, :3]
        qacc = torch.as_tensor(mjd.qacc.numpy(), device=u.device)[:, :3] if hasattr(mjd, "qacc") else None
        Fc = tt(sensor.data.net_forces_w).sum(1) if sensor is not None else None
        nacon = int(mjd.nacon.numpy()[0])
        if len(p_hist) == 3 and vi_prev is not None:
            a_pos = (p_hist[2] - 2 * p_hist[1] + p_hist[0]) / dt**2
            a_vel = (vi - vi_prev) / dt
            a_mj = (vm - vm_prev) / dt if (vm is not None and vm_prev is not None) else None
            for e in range(args_cli.envs):
                if bool(dones[e]): continue
                rows.append((float(a_pos[e].norm()) / g, float(a_vel[e].norm()) / g, float(a_mj[e].norm()) / g if a_mj is not None else float('nan'),
                             float(qacc[e].norm()) / g if qacc is not None else float('nan'), float(qc[e].norm()), float(Fc[e].norm()) if Fc is not None else float('nan'), nacon, step, e))
        vi_prev = vi.clone(); vm_prev = vm.clone() if vm is not None else None
rows_by_pos = sorted(rows, key=lambda r: -r[0]); rows_by_vel = sorted(rows, key=lambda r: -r[1])
print(f"[cr] {len(rows)} samples.  columns: a_pos(g)  a_velIsaac(g)  a_mjCOM(g)  qacc_root(g)  |qfrc_constraint_root|(N)  |sensor contact|(N)  ncon  step env")
print("[cr] --- top by POSITION-based COM acceleration (the real one):")
for r in rows_by_pos[:6]: print("[cr]  " + "  ".join(f"{x:8.2f}" if isinstance(x, float) else f"{x:5d}" for x in r))
print("[cr] --- top by Isaac VELOCITY-based COM acceleration:")
for r in rows_by_vel[:6]: print("[cr]  " + "  ".join(f"{x:8.2f}" if isinstance(x, float) else f"{x:5d}" for x in r))
import statistics
print(f"[cr] median a_pos={statistics.median(r[0] for r in rows):.2f}g  median a_velIsaac={statistics.median(r[1] for r in rows):.2f}g  frac samples with qfrc_constraint_root==0: {sum(1 for r in rows if r[4]==0)/len(rows):.2f}  frac with sensor contact>50N: {sum(1 for r in rows if r[5]>50)/len(rows):.2f}")
env.close(); simulation_app.close()
