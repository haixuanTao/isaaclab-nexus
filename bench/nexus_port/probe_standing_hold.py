"""Physics-only stability test: zero policy actions (= default joint targets through AGILE's
DelayedDCMotor actuators) from STANDING resets. Does the G1 stay up? Nexus vs PhysX.
usage: probe_standing_hold.py [num_envs] [control_steps]   (NEXUS_BACKEND=physx for the stock env)"""
import os, sys, torch, numpy as np
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_nexus.envs import nexusify
N = int(sys.argv[1]) if len(sys.argv) > 1 else 256; STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 250
TASK = "HeightTracking-G1-v0"; G1 = os.environ.get("NEXUS_G1_MJCF", "/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"); BACKEND = os.environ.get("NEXUS_BACKEND", "nexus")
cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); cfg.scene.num_envs = N; cfg.seed = 42
cfg.events.reset_base.params["standing_ratio"] = 1.0 if "standing_ratio" in getattr(cfg.events.reset_base, "params", {}) else None  # all standing
for name in ("push_robot", "randomize_physics_material", "randomize_base_com"):            # no pushes / DR
    if getattr(cfg.events, name, None) is not None: setattr(cfg.events, name, None)
if BACKEND == "nexus": nexusify(cfg, G1)
env = gym.make(TASK, cfg=cfg).unwrapped; env.reset()          # no pre_learn -> dataset absent -> resets are STANDING
robot = env.scene.articulations["robot"]; act = torch.zeros(N, env.action_manager.total_action_dim, device="cuda:0")
z0 = robot.data.root_link_pos_w.torch[:, 2].clone(); hist = []
for i in range(STEPS):
    env.step(act); z = robot.data.root_link_pos_w.torch[:, 2]
    if i in (24, 49, 99, 149, 199, STEPS - 1): hist.append((i + 1, float(z.mean()), float((z > 0.6).float().mean())))
print(f"backend={BACKEND} N={N} | z at reset mean {z0.mean():.2f} | " + " | ".join(f"t={s/50:.1f}s z {m:.2f} up {u*100:.0f}%" for s, m, u in hist))
env.close(); app.close()
