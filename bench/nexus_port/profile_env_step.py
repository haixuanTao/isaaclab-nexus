"""Where the Nexus-backed AGILE env step spends its time.

usage: profile_env_step.py [num_envs] [control_steps]
Times, per control step: physics (NexusManager.step), articulation write+update,
sensor updates, and the rest of ManagerBasedRLEnv.step (obs/reward/termination).
"""
import os, sys, time
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym, torch
import agile.rl_env.tasks  # noqa: registers tasks
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_nexus.envs import nexusify
from isaaclab_nexus.physics.nexus_manager import NexusManager

TASK = "HeightTracking-G1-v0"; G1 = "/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
REDUCE = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True
SOLVER_IT = int(sys.argv[4]) if len(sys.argv) > 4 else 4

cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
cfg.scene.num_envs = N; cfg.seed = 42
nexusify(cfg, G1, contact_reduction=REDUCE, solver_iterations=SOLVER_IT)
env = gym.make(TASK, cfg=cfg).unwrapped
scene = env.scene
robot = scene.articulations["robot"]

acc = {k: 0.0 for k in ("physics", "art_write", "art_update", "sensors", "other")}
def timed(fn, key):
    def wrap(*a, **kw):
        torch.cuda.synchronize(); t = time.perf_counter()
        out = fn(*a, **kw)
        torch.cuda.synchronize(); acc[key] += time.perf_counter() - t
        return out
    return wrap

NexusManager.step = staticmethod(timed(NexusManager.step, "physics"))   # already bound to the class
robot.write_data_to_sim = timed(robot.write_data_to_sim, "art_write")
robot.update = timed(robot.update, "art_update")
for s in scene.sensors.values():
    s.update = timed(s.update, "sensors")

env.reset()
torch.cuda.synchronize(); t0 = time.perf_counter()
act = torch.zeros(N, env.action_manager.total_action_dim, device="cuda:0")
for i in range(STEPS):
    env.step(act)
torch.cuda.synchronize(); total = time.perf_counter() - t0
acc["other"] = total - sum(v for k, v in acc.items() if k != "other")

print(f"\n=== Nexus backend env-step breakdown | {N} envs x {STEPS} control steps (decimation {env.cfg.decimation}) ===")
print(f"{'section':<12} {'total_s':>9} {'ms/ctrl-step':>13} {'share':>7}")
for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    print(f"{k:<12} {v:9.3f} {v/STEPS*1e3:13.2f} {v/total*100:6.1f}%")
print(f"{'TOTAL':<12} {total:9.3f} {total/STEPS*1e3:13.2f} {100.0:6.1f}%")
print(f"env-steps/s: {N*STEPS/total:.0f} | contact_reduction={REDUCE} | solver_iterations={SOLVER_IT}")
# how far the feet sit below the terrain surface they stand on
feet_ids = robot.find_bodies(".*ankle_roll_link")[0]
p_w = robot.data.body_link_pos_w.torch[:, feet_ids]
gap = p_w[..., 2].min(1).values - scene.terrain.heights_at(p_w[..., :2].mean(1))
q = torch.tensor([0.05, 0.5, 0.95], device=gap.device)
print("foot - terrain gap (m) p05/p50/p95:", [round(v, 3) for v in torch.quantile(gap, q).tolist()])
env.close(); app.close()
