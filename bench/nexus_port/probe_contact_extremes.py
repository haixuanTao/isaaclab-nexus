"""Max contact-sensor force and max |critic obs| at training scale for a checkpoint.
usage: probe_contact_extremes.py <ckpt> [num_envs] [steps]   (NEXUS_SOLVER_ITERS selects substeps)"""
import os, sys, torch, numpy as np
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym, importlib
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
from agile.rl_env.rsl_rl import RslRlVecEnvWrapper, make_rsl_rl_runner
from isaaclab_nexus.envs import nexusify
CK = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 300
TASK = "HeightTracking-G1-v0"; G1 = os.environ.get("NEXUS_G1_MJCF", "/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"); S = int(os.environ.get("NEXUS_SOLVER_ITERS", "1"))
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
if os.environ.get("NEXUS_EMP_NORM", "1") != "0": agent_cfg.empirical_normalization = True   # match train_nexus.py (checkpoints carry normalizer buffers)
env_cfg.scene.num_envs = N; env_cfg.seed = 42
BACKEND = os.environ.get("NEXUS_BACKEND", "nexus")
if BACKEND == "nexus": nexusify(env_cfg, G1, solver_iterations=S)
env = gym.make(TASK, cfg=env_cfg); base = env.unwrapped
pre = gym.spec(TASK).kwargs.get("pre_learn_entry_point"); mod, fn = pre.split(":"); getattr(importlib.import_module(mod), fn)(base, TASK, agent_cfg); base.reset()
wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = make_rsl_rl_runner(wenv, agent_cfg, log_dir=None, device=agent_cfg.device); runner.load(CK); policy = runner.get_inference_policy(device=agent_cfg.device)
obs = wenv.get_observations(); cs = base.scene.sensors["contact_forces"]
fmax, f999, cmax, nonfinite = [], [], [], 0
with torch.inference_mode():
    for i in range(STEPS):
        obs, _, _, _ = wenv.step(policy(obs))
        f = cs.data.net_forces_w.torch.norm(dim=-1)                   # (N, bodies)
        fmax.append(float(f.max())); f999.append(float(torch.quantile(f.flatten().float()[::7], 0.999)))
        og = base.observation_manager.compute()                        # dict of groups (recompute; inference only)
        c = og["critic"] if "critic" in og else next(v for k, v in og.items() if "critic" in str(k))
        c = c if torch.is_tensor(c) else torch.cat([t.flatten(1) for t in c.values()], 1)
        cmax.append(float(c.abs().max())); nonfinite += int((~torch.isfinite(c)).sum())
fmax, f999, cmax = map(np.array, (fmax, f999, cmax))
print(f"backend={BACKEND} solver_iters={S if BACKEND=='nexus' else 'physx'} N={N} steps={STEPS} | contact force N: max {fmax.max():.0f}, median-of-step-max {np.median(fmax):.0f}, p99.9 (median over steps) {np.median(f999):.0f} | robot weight 346 N "
      f"| critic obs: max |value| {cmax.max():.1f} (median-of-step-max {np.median(cmax):.1f}), non-finite {nonfinite}")
env.close(); app.close()
