"""Train AGILE's HeightTracking-G1-v0 with rsl_rl on the Nexus backend (mirrors scripts/train.py without hydra).
usage: train_nexus.py [num_envs] [iterations]"""
import os, sys, time
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym, torch
import agile.rl_env.tasks  # noqa: registers tasks + patches
from isaaclab_tasks.utils import load_cfg_from_registry
from agile.rl_env.rsl_rl import RslRlVecEnvWrapper, make_rsl_rl_runner
from isaaclab_nexus.envs import nexusify
TASK = "HeightTracking-G1-v0"; G1 = "/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
NENV = int(sys.argv[1]) if len(sys.argv) > 1 else 1024; ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
env_cfg.scene.num_envs = NENV; env_cfg.seed = agent_cfg.seed; agent_cfg.max_iterations = ITERS; agent_cfg.run_name = "nexus"
nexusify(env_cfg, G1)
t0 = time.perf_counter(); env = gym.make(TASK, cfg=env_cfg); print(f"[nexus] env built in {time.perf_counter()-t0:.1f}s")
pre = gym.spec(TASK).kwargs.get("pre_learn_entry_point")
if pre:
    import importlib; mod, fn = pre.split(":"); t0 = time.perf_counter(); getattr(importlib.import_module(mod), fn)(env.unwrapped, TASK, agent_cfg); print(f"[nexus] pre_learn (fallen-state dataset) done in {time.perf_counter()-t0:.1f}s")
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
log_dir = os.path.abspath(os.path.join("/workspace/bench/nexus_port/logs", time.strftime("%Y-%m-%d_%H-%M-%S") + "_nexus"))
runner = make_rsl_rl_runner(env, agent_cfg, log_dir=log_dir, device=agent_cfg.device)
t0 = time.perf_counter(); runner.learn(num_learning_iterations=ITERS, init_at_random_ep_len=True); el = time.perf_counter() - t0
steps = ITERS * agent_cfg.num_steps_per_env * NENV
print(f"[nexus] {ITERS} PPO iterations x {agent_cfg.num_steps_per_env} steps x {NENV} envs in {el:.1f}s -> {el/ITERS*1000:.0f} ms/iter, {steps/el:.0f} env-steps/s | log {log_dir}")
print("TRAIN ON NEXUS OK"); env.close(); app.close()
