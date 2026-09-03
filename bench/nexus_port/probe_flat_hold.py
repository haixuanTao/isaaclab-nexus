"""Standing hold on a FLAT floor (no terrain), Nexus. Zero actions, default targets, gain 1.
If the G1 stands here but not on the terrain tile, the problem is the terrain contact / spawn
height on rough tiles; if it topples here too, it is the foot contact model itself."""
import os, sys, torch
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_nexus.envs import nexusify
TASK = "HeightTracking-G1-v0"; G1 = os.environ.get("NEXUS_G1_MJCF", "/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"); N = 128
FRICTION = float(os.environ.get("NEXUS_FLAT_FRICTION", "1.0"))
cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); cfg.scene.num_envs = N; cfg.seed = 42
for name in ("push_robot", "randomize_physics_material", "randomize_base_com"):
    if getattr(cfg.events, name, None) is not None: setattr(cfg.events, name, None)
nexusify(cfg, G1)
cfg.scene.terrain = None                                   # no tile: the loader adds a flat cuboid floor under the robot
cfg.scene.robot.spawn.auto_floor = True
cfg.scene.height_measurement_sensor = None                 # ray caster needs a terrain
if hasattr(cfg.observations.policy, "height_scan"): cfg.observations.policy.height_scan = None
for grp in ("policy", "critic"):
    g = getattr(cfg.observations, grp, None)
    for t in ("base_height", "height_scan"):
        if g is not None and getattr(g, t, None) is not None and "height_measurement_sensor" in str(getattr(getattr(g, t), "params", {})): setattr(g, t, None)
env = gym.make(TASK, cfg=cfg).unwrapped; env.reset(); robot = env.scene.articulations["robot"]
act = torch.zeros(N, env.action_manager.total_action_dim, device="cuda:0"); feet = robot.find_bodies(".*ankle_roll_link")[0]
fz0 = robot.data.body_link_pos_w.torch[:, feet, 2]; out = []
for i in range(150):
    env.step(act); z = robot.data.root_link_pos_w.torch[:, 2]
    if i in (24, 49, 99, 149): out.append(f"t={(i+1)/50:.1f}s z {float(z.mean()):.2f} up {float((z>0.6).float().mean())*100:.0f}%")
print(f"[nexus FLAT floor, friction {FRICTION}] foot z at reset min/median {float(fz0.min()):.3f}/{float(fz0.median()):.3f} | " + " | ".join(out))
env.close(); app.close()
