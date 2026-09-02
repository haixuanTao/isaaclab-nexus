"""A/B: replay one physics step from a CUDA graph vs re-encoding it every step.

usage: test_cuda_graph.py <cuda_graph_warmup>   (0 = off)
Steps the real AGILE env, times the second half only, and prints a state
fingerprint so the two runs can be checked against each other.
"""
import sys, time
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym, torch
import agile.rl_env.tasks  # noqa: registers tasks
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_nexus.envs import nexusify
from isaaclab_nexus.physics.nexus_manager import NexusManager

TASK = "HeightTracking-G1-v0"; G1 = "/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
WARMUP = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N, STEPS = 4096, 60                                   # 60 control steps = 240 physics steps

cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
cfg.scene.num_envs = N; cfg.seed = 42
nexusify(cfg, G1, cuda_graph_warmup=WARMUP)
env = gym.make(TASK, cfg=cfg).unwrapped
env.reset()
act = torch.zeros(N, env.action_manager.total_action_dim, device="cuda:0")

for i in range(STEPS // 2):                            # warm/settle half
    env.step(act)
torch.cuda.synchronize(); t0 = time.perf_counter()
for i in range(STEPS // 2):
    env.step(act)
torch.cuda.synchronize(); el = time.perf_counter() - t0

d = env.scene.articulations["robot"].data
z = d.root_link_pos_w.torch[:, 2]; q = d.joint_pos.torch
print(f"cuda_graph_warmup={WARMUP} | graph active: {NexusManager._graph} | "
      f"{STEPS//2} control steps in {el:.2f}s -> {el/(STEPS//2)*1e3:.1f} ms/step, {N*(STEPS//2)/el:.0f} env-steps/s")
print(f"  fingerprint: root z mean {z.mean():.6f} std {z.std():.6f} | joint_pos abs-sum {q.abs().sum():.4f}")
env.close(); app.close()
