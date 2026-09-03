"""Training-scale outlier probe: roll a checkpoint on N envs with training-style resets and log
per-step extremes of reward and physics, naming the env and the reward terms responsible.
usage: probe_outliers.py <ckpt> [num_envs] [control_steps]"""
import os, sys, torch, numpy as np
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym, importlib
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
from agile.rl_env.rsl_rl import RslRlVecEnvWrapper, make_rsl_rl_runner
from isaaclab_nexus.envs import nexusify
CK = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 600
TASK = "HeightTracking-G1-v0"; G1 = os.environ.get("NEXUS_G1_MJCF", "/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml")
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
if os.environ.get("NEXUS_EMP_NORM", "1") != "0": agent_cfg.empirical_normalization = True   # match train_nexus.py (checkpoints carry normalizer buffers)
env_cfg.scene.num_envs = N; env_cfg.seed = 42
nexusify(env_cfg, G1, solver_iterations=int(os.environ.get("NEXUS_SOLVER_ITERS", "1")), agent_cfg=agent_cfg)
env = gym.make(TASK, cfg=env_cfg); base = env.unwrapped
pre = gym.spec(TASK).kwargs.get("pre_learn_entry_point"); mod, fn = pre.split(":"); getattr(importlib.import_module(mod), fn)(base, TASK, agent_cfg); base.reset()
wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = make_rsl_rl_runner(wenv, agent_cfg, log_dir=None, device=agent_cfg.device); runner.load(CK); policy = runner.get_inference_policy(device=agent_cfg.device)
obs = wenv.get_observations(); robot = base.scene.articulations["robot"]; rm = base.reward_manager
terms = list(rm.active_terms)
worst = []   # (reward, step, env, per-term dict, jv, rv, z)
rmin_hist, jv_hist, rv_hist = [], [], []
with torch.inference_mode():
    for i in range(STEPS):
        obs, rew, dones, _ = wenv.step(policy(obs))
        d = robot.data
        jv = d.joint_vel.torch.abs().amax(1); rv = d.root_lin_vel_w.torch.norm(dim=-1); z = d.root_link_pos_w.torch[:, 2]
        e = int(torch.argmin(rew)); rmin_hist.append(float(rew[e])); jv_hist.append(float(jv.max())); rv_hist.append(float(rv.max()))
        if len(worst) < 8 or float(rew[e]) < worst[-1][0]:
            per = {}
            sr = getattr(rm, "_step_reward", None)
            if sr is not None:
                for k, name in enumerate(terms): per[name] = float(sr[e, k])
            worst.append((float(rew[e]), i, e, per, float(jv[e]), float(rv[e]), float(z[e]))); worst.sort(key=lambda w: w[0]); worst = worst[:8]
rmin = np.array(rmin_hist); print(f"N={N} steps={STEPS} solver_iters={os.environ.get('NEXUS_SOLVER_ITERS','1')} | per-step min reward: median {np.median(rmin):.2f}  p1 {np.percentile(rmin,1):.2f}  min {rmin.min():.2f} | max |joint_vel| over run {max(jv_hist):.1f} rad/s | max root speed {max(rv_hist):.2f} m/s")
print("worst single (env, step) rewards and what made them:")
for r, i, e, per, jv, rv, z in worst[:5]:
    top = sorted(per.items(), key=lambda kv: kv[1])[:4]
    print(f"  reward {r:9.2f} step {i:4d} env {e:4d} | jv {jv:5.1f} rv {rv:4.2f} z {z:5.2f} | " + ", ".join(f"{k}={v:.2f}" for k, v in top))
env.close(); app.close()
