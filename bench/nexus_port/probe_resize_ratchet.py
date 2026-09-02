"""Roll a checkpoint and watch the engine's collision-buffer ratchet.
usage: probe_resize_ratchet.py <ckpt> <policy grow|fit|fixed> [num_envs] [control_steps]
Prints rbd_resize_stats() every 200 control steps + step timing; the backend warns on any ratchet."""
import os, sys, time, warnings
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym, torch
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
from agile.rl_env.rsl_rl import RslRlVecEnvWrapper, make_rsl_rl_runner
from isaaclab_nexus.envs import nexusify
from isaaclab_nexus.physics.nexus_manager import NexusManager
warnings.simplefilter("always")
CKPT, POLICY = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 2048; STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 1200
TASK = "HeightTracking-G1-v0"; G1 = os.environ.get("NEXUS_G1_MJCF", "/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml")
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
env_cfg.scene.num_envs = N; env_cfg.seed = 3
nexusify(env_cfg, G1, collisions_resize_policy=POLICY)
env = gym.make(TASK, cfg=env_cfg); wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = make_rsl_rl_runner(wenv, agent_cfg, log_dir=None, device=agent_cfg.device); runner.load(CKPT); policy = runner.get_inference_policy(device=agent_cfg.device)
obs = wenv.get_observations(); st = NexusManager._state
torch.cuda.synchronize(); t0 = time.perf_counter(); peak = 0
with torch.inference_mode():
    for i in range(1, STEPS + 1):
        obs, _, _, _ = wenv.step(policy(obs))
        if i % 200 == 0:
            torch.cuda.synchronize(); el = time.perf_counter() - t0; t0 = time.perf_counter()
            s = st.rbd_resize_stats(); peak = max(peak, int(s.get("pairs_len", 0)))
            print(f"[{POLICY}] ctrl step {i:5d} | {el/200*1e3:6.1f} ms/step | pairs_len {s.get('pairs_len')} (peak {peak}) "
                  f"cap/batch {s.get('capacity_per_batch')} min {s.get('capacity_min')} | max_colors {s.get('max_colors')} hw {s.get('colors_high_water')} inert {s.get('rb_contacts_inert')}", flush=True)
print(f"[{POLICY}] DONE peak pairs/batch {peak}"); env.close(); app.close()
