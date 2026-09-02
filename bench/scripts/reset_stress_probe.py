#!/usr/bin/env python3
"""Is the rare blow-up tied to resets? Force resets of a random 10% of envs every
control step (via the env's own _reset_idx), random actions, and check joint
state after each step. Reports the first non-finite with the same joint-level
breakdown as the watchdog (imported from it)."""
import argparse, sys, os
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--envs", type=int, default=2048); parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--reset_frac", type=float, default=0.10)
parser.add_argument("--reset_write", type=str, default="none", help="none | rest | dataset | dataset_sanitized")
parser.add_argument("--dataset", type=str, default="/workspace/WBC-AGILE-NEWTON/fallen_states_cache/fallen_states_v6_HeightTracking_G1_v0_flat_edd3202a.pt")
AppLauncher.add_app_launcher_args(parser); args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, numpy as np, torch
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from agile.rl_env.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.envs); env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped
env = RslRlVecEnvWrapper(env)   # so the watchdog hook (if armed) sees every step
obs, _ = env.reset(); torch.manual_seed(1)
robot = u.scene["robot"]
def tt(x): return x.torch if hasattr(x, "torch") else x
print(f"\n[rs] envs={args_cli.envs} reset_frac={args_cli.reset_frac} steps={args_cli.steps} watchdog={os.environ.get('AGILE_NAN_WATCHDOG')}", flush=True)
n_resets = 0
DS = None
if args_cli.reset_write.startswith("dataset"):
    d = torch.load(args_cli.dataset, map_location="cpu", weights_only=False)
    st = d["states_by_level"][0]
    names_ds = d.get("joint_names"); names_rb = list(robot.joint_names)
    perm = [names_ds.index(n) for n in names_rb] if names_ds else list(range(len(names_rb)))
    DS = {"joint_pos": st["joint_pos"][:, perm].contiguous(), "joint_vel": st["joint_vel"][:, perm].contiguous(), "root_pos_rel": st["root_pos_rel"], "root_quat": st["root_quat"]}
    print(f"[rs] dataset {args_cli.dataset.rsplit('/',1)[-1]}: {DS['joint_pos'].shape[0]} states, max|joint_vel|={DS['joint_vel'].abs().max():.1f}", flush=True)
for step in range(args_cli.steps):
    a = torch.randn(u.action_space.shape, device=u.device)
    obs, rew, dones, extras = env.step(a)
    ids = torch.nonzero(torch.rand(args_cli.envs, device=u.device) < args_cli.reset_frac).flatten()
    if ids.numel():
        from isaaclab_newton.physics import NewtonManager
        e = int(ids[0]); m = NewtonManager.get_model(); nd = m.joint_dof_count // args_cli.envs
        pre_qd = float(np.abs(NewtonManager._state_0.joint_qd.numpy()[e*nd:(e+1)*nd]).max()) if step % 50 == 0 else None
        u._reset_idx(ids); n_resets += int(ids.numel())
        if args_cli.reset_write != "none":
            if args_cli.reset_write == "rest":
                jp = tt(robot.data.default_joint_pos)[ids].clone(); jv = torch.zeros_like(jp)
            else:
                k = torch.randint(0, DS["joint_pos"].shape[0], (ids.numel(),))
                jp = DS["joint_pos"][k].to(u.device); jv = DS["joint_vel"][k].to(u.device)
                if args_cli.reset_write == "dataset_sanitized":
                    lim = tt(robot.data.joint_pos_limits)[ids]           # (n, J, 2)
                    jp = torch.minimum(torch.maximum(jp, lim[..., 0]), lim[..., 1]); jv = torch.zeros_like(jv)
            # a real reset also places the root: default spawn for 'rest', the dataset's own pose otherwise
            if args_cli.reset_write == "rest":
                rp = tt(robot.data.default_root_state)[ids, :7].clone(); rp[:, :3] += u.scene.env_origins[ids]
            else:
                rp = torch.cat([DS["root_pos_rel"][k].to(u.device) + u.scene.env_origins[ids], DS["root_quat"][k].to(u.device)], dim=-1)
            robot.write_root_pose_to_sim(rp, env_ids=ids)
            robot.write_root_velocity_to_sim(torch.zeros(ids.numel(), 6, device=u.device), env_ids=ids)
            robot.write_joint_state_to_sim(jp, jv, env_ids=ids)
        if step % 50 == 0:
            post_qd = NewtonManager._state_0.joint_qd.numpy()[e*nd:(e+1)*nd]
            bind = robot.data._sim_bind_joint_vel if hasattr(robot.data, "_sim_bind_joint_vel") else None
            bind_qd = float(np.abs(bind.numpy()[e]).max()) if bind is not None else float('nan')
            print(f"[rs] reset env {e}: newton max|qd| before={pre_qd:.3g} after={float(np.abs(post_qd).max()):.3g}  isaac sim-bind joint_vel after={bind_qd:.3g}  "
                  f"(state_0 id={id(NewtonManager._state_0)})", flush=True)
    jv = tt(robot.data.joint_vel); jp = tt(robot.data.joint_pos)
    if not (torch.isfinite(jv).all() and torch.isfinite(jp).all()):
        bad = (~torch.isfinite(jv)).any(dim=1) | (~torch.isfinite(jp)).any(dim=1)
        print(f"[rs] NON-FINITE joint state at step {step} in {int(bad.sum())} envs (first {int(bad.nonzero()[0])}) after {n_resets} forced resets", flush=True)
        break
    if step % 250 == 0:
        print(f"[rs] step {step}: max|joint_vel|={jv.abs().max().item():.1f}  max|joint_pos|={jp.abs().max().item():.2f}  resets so far={n_resets}", flush=True)
else:
    print(f"[rs] SURVIVED {args_cli.steps} steps with {n_resets} forced resets", flush=True)
env.close(); simulation_app.close()
