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
TASK = "HeightTracking-G1-v0"; G1 = os.environ.get("NEXUS_G1_MJCF", "/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml")
NENV = int(sys.argv[1]) if len(sys.argv) > 1 else 1024; ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
# Empirical observation normalization ON by default on this backend (AGILE ships it off). Without it
# every run collapsed at ~4,000 iterations (critic divergence); with it, v10 trained through cleanly.
# NEXUS_EMP_NORM=0 restores AGILE's setting.
if os.environ.get("NEXUS_EMP_NORM", "1") != "0": agent_cfg.empirical_normalization = True; print("[nexus] empirical observation normalization ON (actor + critic) -- backend default, deviation from AGILE's cfg")
env_cfg.scene.num_envs = NENV; env_cfg.seed = int(os.environ.get("NEXUS_SEED", agent_cfg.seed)); agent_cfg.seed = env_cfg.seed; agent_cfg.max_iterations = ITERS; agent_cfg.run_name = "nexus"
nexusify(env_cfg, G1, critic_force_clip_n=(None if os.environ.get("NEXUS_FORCE_CLIP", "5000") in ("0", "none", "None") else float(os.environ.get("NEXUS_FORCE_CLIP", "5000"))))
FCLIP = float(os.environ.get("NEXUS_DIAG_FORCE_CLIP", "0") or 0)        # override of nexusify's critic_force_clip_n (default 5000 N)
if FCLIP:
    env_cfg.observations.critic.contact_forces.clip = (-FCLIP, FCLIP); print(f"[nexus] DIAGNOSTIC critic contact_forces clip = ±{FCLIP} N")
if os.environ.get("NEXUS_DIAG_NO_FORCE_OBS") == "1":                     # DIAGNOSTIC: drop the critic's contact-force term entirely
    # zero the term's SCALE rather than removing it: keeps the critic input width so a checkpoint still loads
    env_cfg.observations.critic.contact_forces.scale = 0.0; print("[nexus] DIAGNOSTIC critic contact_forces observation ZEROED (scale=0)")
t0 = time.perf_counter(); env = gym.make(TASK, cfg=env_cfg); print(f"[nexus] env built in {time.perf_counter()-t0:.1f}s")
pre = gym.spec(TASK).kwargs.get("pre_learn_entry_point")
if pre:
    import importlib; mod, fn = pre.split(":"); t0 = time.perf_counter(); getattr(importlib.import_module(mod), fn)(env.unwrapped, TASK, agent_cfg); print(f"[nexus] pre_learn (fallen-state dataset) done in {time.perf_counter()-t0:.1f}s")
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
log_dir = os.path.abspath(os.path.join("/workspace/bench/nexus_port/logs", time.strftime("%Y-%m-%d_%H-%M-%S") + "_nexus"))
runner = make_rsl_rl_runner(env, agent_cfg, log_dir=log_dir, device=agent_cfg.device)
RESUME = os.environ.get("NEXUS_RESUME")                                   # checkpoint to continue from
if RESUME:
    runner.load(RESUME, strict=(os.environ.get("NEXUS_EMP_NORM", "1") == "0")); print(f"[nexus] resumed from {RESUME} at iteration {runner.current_learning_iteration}")
t0 = time.perf_counter()
CHUNK = int(os.environ.get("NEXUS_STATS_EVERY", "0") or 0)          # >0: learn in chunks and log allocator/engine stats between them
if CHUNK:
    from isaaclab_nexus.physics.nexus_manager import NexusManager
    done = 0
    while done < ITERS:
        n = min(CHUNK, ITERS - done); runner.learn(num_learning_iterations=n, init_at_random_ep_len=(done == 0)); done += n
        ms = torch.cuda.memory_stats(); st = NexusManager._state.rbd_resize_stats() if hasattr(NexusManager._state, "rbd_resize_stats") else {}
        print(f"[nexus-stats] iter {done} | {(time.perf_counter()-t0)/done*1000:.0f} ms/iter avg | torch alloc {ms['allocated_bytes.all.current']/2**30:.2f} GiB "
              f"reserved {ms['reserved_bytes.all.current']/2**30:.2f} GiB retries {ms['num_alloc_retries']} ooms {ms['num_ooms']} "
              f"| engine cap/batch {st.get('capacity_per_batch')} pairs {st.get('pairs_len')} max_colors {st.get('max_colors')} peak_pairs {NexusManager._pairs_peak}", flush=True)
else:
    runner.learn(num_learning_iterations=ITERS, init_at_random_ep_len=True)
el = time.perf_counter() - t0
steps = ITERS * agent_cfg.num_steps_per_env * NENV
print(f"[nexus] {ITERS} PPO iterations x {agent_cfg.num_steps_per_env} steps x {NENV} envs in {el:.1f}s -> {el/ITERS*1000:.0f} ms/iter, {steps/el:.0f} env-steps/s | log {log_dir}")
print("TRAIN ON NEXUS OK"); env.close(); app.close()
