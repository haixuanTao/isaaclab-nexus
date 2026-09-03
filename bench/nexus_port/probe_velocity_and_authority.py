"""(1) Zero-g: is joint_vel consistent with d(joint_pos)/dt, and what torque does the actuator apply?
(2) Gravity on, standing resets, zero actions, actuator stiffness/damping scaled by GAIN: does it stand?
usage: probe_velocity_and_authority.py [gain_scale]"""
import os, sys, torch
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_nexus.envs import nexusify
from isaaclab_nexus.physics.nexus_manager import NexusManager
GAIN = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
TASK = "HeightTracking-G1-v0"; G1 = os.environ.get("NEXUS_G1_MJCF", "/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml")
def make(gravity, N):
    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); cfg.scene.num_envs = N; cfg.seed = 42; cfg.sim.gravity = gravity
    for name in ("push_robot", "randomize_physics_material", "randomize_base_com"):
        if getattr(cfg.events, name, None) is not None: setattr(cfg.events, name, None)
    if GAIN != 1.0:
        for a in cfg.scene.robot.actuators.values():
            for k in ("stiffness", "damping"):
                v = getattr(a, k); setattr(a, k, {kk: vv * GAIN for kk, vv in v.items()} if isinstance(v, dict) else v * GAIN)
    nexusify(cfg, G1); env = gym.make(TASK, cfg=cfg).unwrapped; env.reset(); return env
# ---- (1) velocity channel, zero-g, robot in the air, step the left knee target
env = make((0.0, 0.0, 0.0), 8); robot = env.scene.articulations["robot"]; dt = env.physics_dt; j = robot.joint_names.index("left_knee_joint")
pose = robot.data.default_root_pose.torch.clone(); pose[:, 2] = 3.0; robot.write_root_pose_to_sim(pose)
q0 = robot.data.default_joint_pos.torch.clone(); robot.write_joint_state_to_sim(q0, torch.zeros_like(q0)); tgt = q0.clone(); tgt[:, j] += 0.3
prev = robot.data.joint_pos.torch[0, j].item(); print("step   q        v_reported   dq/dt      torque")
for i in range(12):
    robot.set_joint_position_target(tgt); robot.write_data_to_sim(); NexusManager.step(); robot.update(dt)
    q = robot.data.joint_pos.torch[0, j].item(); v = robot.data.joint_vel.torch[0, j].item(); tq = robot.data.applied_torque.torch[0, j].item()
    print(f"{i:4d}  {q:+.4f}   {v:+9.3f}   {(q-prev)/dt:+9.3f}   {tq:+8.2f}"); prev = q
env.close()
# ---- (2) authority: gravity on, standing resets, zero actions
env = make((0.0, 0.0, -9.81), 256); robot = env.scene.articulations["robot"]; act = torch.zeros(256, env.action_manager.total_action_dim, device="cuda:0")
out = []
for i in range(150):
    env.step(act); z = robot.data.root_link_pos_w.torch[:, 2]
    if i in (24, 49, 99, 149): out.append(f"t={(i+1)/50:.1f}s z {float(z.mean()):.2f} up {float((z>0.6).float().mean())*100:.0f}%")
tq = robot.data.applied_torque.torch.abs(); print(f"gain x{GAIN}: " + " | ".join(out) + f" | |torque| median {float(tq.median()):.1f} max {float(tq.max()):.1f} N·m")
env.close(); app.close()
