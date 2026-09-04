"""Per-body mass and inertia (diagonal, body frame) as each backend loads them. NEXUS_BACKEND=physx for the stock env."""
import os, torch
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
BACKEND = os.environ.get("NEXUS_BACKEND", "nexus"); TASK = "HeightTracking-G1-v0"
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point"); env_cfg.scene.num_envs = 2
for k in list(vars(env_cfg.events).keys()):
    if "com" in k or "mass" in k or "material" in k: setattr(env_cfg.events, k, None)
if BACKEND == "nexus":
    from isaaclab_nexus.envs import nexusify; nexusify(env_cfg, "/workspace/bench/nexus_port/g1_29dof_convex64.xml", agent_cfg=agent_cfg)
env = gym.make(TASK, cfg=env_cfg); r = env.unwrapped.scene.articulations["robot"]
def T(x): return x.torch if hasattr(x, "torch") else x
m = T(r.data.default_mass)[0]; I = T(r.data.default_inertia)[0].reshape(-1, 3, 3)
print(f"[{BACKEND}] total mass {float(m.sum()):.2f} kg | " + " ; ".join(f"{n}: m {float(mm):.2f} Ixx/Iyy/Izz {float(ii[0,0]):.4f}/{float(ii[1,1]):.4f}/{float(ii[2,2]):.4f}" for n, mm, ii in zip(r.body_names, m, I) if n in ("pelvis", "torso_link", "left_hip_pitch_link", "left_knee_link", "left_ankle_roll_link", "left_shoulder_pitch_link", "left_wrist_yaw_link")))
env.close(); app.close()
