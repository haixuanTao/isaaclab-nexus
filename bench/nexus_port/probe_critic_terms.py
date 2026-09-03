"""Per-term extremes of the CRITIC observation group at training scale, Nexus vs PhysX.
usage: probe_critic_terms.py <ckpt> [num_envs] [steps]   (NEXUS_BACKEND=physx for the stock env)"""
import os, sys, torch, numpy as np
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym, importlib
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
from agile.rl_env.rsl_rl import RslRlVecEnvWrapper, make_rsl_rl_runner
from isaaclab_nexus.envs import nexusify
CK = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 2048; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
TASK = "HeightTracking-G1-v0"; G1 = os.environ.get("NEXUS_G1_MJCF", "/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"); BACKEND = os.environ.get("NEXUS_BACKEND", "nexus")
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
if os.environ.get("NEXUS_EMP_NORM", "1") != "0": agent_cfg.empirical_normalization = True   # match train_nexus.py (checkpoints carry normalizer buffers)
env_cfg.scene.num_envs = N; env_cfg.seed = 42
if BACKEND == "nexus": nexusify(env_cfg, G1, critic_force_clip_n=None)      # AGILE's own critic cfg, for a like-for-like read
env = gym.make(TASK, cfg=env_cfg); base = env.unwrapped
pre = gym.spec(TASK).kwargs.get("pre_learn_entry_point"); mod, fn = pre.split(":"); getattr(importlib.import_module(mod), fn)(base, TASK, agent_cfg); base.reset()
wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = make_rsl_rl_runner(wenv, agent_cfg, log_dir=None, device=agent_cfg.device); runner.load(CK); policy = runner.get_inference_policy(device=agent_cfg.device)
obs = wenv.get_observations(); om = base.observation_manager
names = om.active_terms["critic"]; dims = [int(np.prod(d)) for d in om.group_obs_term_dim["critic"]]
mx = {n: 0.0 for n in names}; p999 = {n: [] for n in names}
with torch.inference_mode():
    for i in range(STEPS):
        obs, _, _, _ = wenv.step(policy(obs))
        c = om.compute()["critic"]; c = c if torch.is_tensor(c) else torch.cat([t.flatten(1) for t in c.values()], 1)
        o = 0
        for n, d in zip(names, dims):
            v = c[:, o:o + d].abs(); o += d
            mx[n] = max(mx[n], float(v.max())); p999[n].append(float(torch.quantile(v.flatten()[::max(1, v.numel() // 200000)], 0.999)))
print(f"backend={BACKEND} N={N} steps={STEPS} | critic terms: max |value| (p99.9 median over steps)")
for n in names: print(f"  {n:<28} max {mx[n]:9.2f}   p99.9 {np.median(p999[n]):8.2f}")
env.close(); app.close()
