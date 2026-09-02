"""AGILE's HeightTracking-G1-v0 (ManagerBasedRLEnv, all managers) on the Nexus backend.
Constructs the env, resets, steps random actions, reports shapes / finiteness / timing."""
import sys, time, torch
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
from isaaclab.envs import ManagerBasedRLEnv
import agile.rl_env  # registers tasks + monkey patches
from agile.rl_env.tasks.stand_up.g1.height_tracking_env_cfg import G1HeightTrackingEnvCfg
from isaaclab_nexus.envs import nexusify
G1 = "/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
NENV = int(sys.argv[1]) if len(sys.argv) > 1 else 64; STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
cfg = G1HeightTrackingEnvCfg(); cfg.scene.num_envs = NENV
nexusify(cfg, G1)
t0 = time.perf_counter(); env = ManagerBasedRLEnv(cfg=cfg); print(f"env built in {time.perf_counter()-t0:.1f}s | scene {env.scene} | obs groups {list(env.observation_manager.group_obs_dim)} | action dim {env.action_manager.total_action_dim}")
obs, _ = env.reset(); print("reset OK | policy obs", {k: tuple(v.shape) for k, v in obs['policy'].items()} if isinstance(obs['policy'], dict) else tuple(obs['policy'].shape))
torch.cuda.synchronize(); t = time.perf_counter(); tot_r = 0.0; dones = 0
for i in range(STEPS):
    act = 0.2 * torch.randn(NENV, env.action_manager.total_action_dim, device=env.device)
    obs, rew, term, trunc, extras = env.step(act); tot_r += rew.mean().item(); dones += int((term | trunc).sum())
torch.cuda.synchronize(); el = time.perf_counter() - t
flat = torch.cat([v.flatten() for v in (obs['policy'].values() if isinstance(obs['policy'], dict) else [obs['policy']])])
print(f"{STEPS} env steps x {NENV} envs: {el:.2f}s -> {STEPS/el:.1f} steps/s, {NENV*STEPS/el:.0f} env-steps/s (x decimation {cfg.decimation} = {NENV*STEPS*cfg.decimation/el:.0f} physics env-steps/s)")
print(f"mean reward/step {tot_r/STEPS:.4f} | episodes ended {dones} | obs finite {bool(torch.isfinite(flat).all())} | reward terms: {list(env.reward_manager.active_terms)[:6]}...")
r = env.scene['robot']; print("root z mean %.3f | torso height cmd mean %.3f" % (r.data.root_pos_w.torch[:, 2].mean(), env.command_manager.get_command('height').mean()))
print("AGILE ENV ON NEXUS OK"); env.close(); app.close()
