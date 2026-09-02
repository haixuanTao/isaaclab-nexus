"""Fidelity check for the shipped config: does the ContactSensor's summed normal
force over ALL bodies equal the robot's weight once it rests on the terrain?
(Explicit Coriolis changes the impulse cadence Zealot divides by; at 1 substep
the backend's scaling must still give ~1.0 x weight.)"""
import os
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym, torch
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_nexus.envs import nexusify
TASK = "HeightTracking-G1-v0"; G1 = os.environ.get("NEXUS_G1_MJCF", "/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml")
N = 512
cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); cfg.scene.num_envs = N; cfg.seed = 42
nexusify(cfg, G1)
env = gym.make(TASK, cfg=cfg).unwrapped; env.reset()
act = torch.zeros(N, env.action_manager.total_action_dim, device="cuda:0")
for _ in range(150):                                   # 3 s: let every robot come to rest
    env.step(act)
robot = env.scene.articulations["robot"]; sensor = env.scene.sensors["contact_forces"]
mass = float(robot.data.default_mass.torch[0].sum()) if hasattr(robot.data, "default_mass") else float("nan")
fz = sensor.data.net_forces_w.torch[..., 2].sum(1)     # (N,) total normal force per env
w = mass * 9.81
v = robot.data.joint_vel.torch.abs().max(1).values
resting = v < 0.05
r = (fz[resting] / w) if resting.any() else fz / w
print(f"robot mass {mass:.2f} kg -> weight {w:.1f} N | resting envs {int(resting.sum())}/{N}")
print(f"sensed normal force / weight: median {r.median():.3f}  p10 {r.quantile(0.1):.3f}  p90 {r.quantile(0.9):.3f}")
env.close(); app.close()
