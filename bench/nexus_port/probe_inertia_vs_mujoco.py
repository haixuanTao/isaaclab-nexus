"""Effective joint inertia: Nexus vs MuJoCo from the same MJCF. Zero-g, robot floating at the
default pose, ONE physics step with a pure generalized torque on one joint (no actuator model),
compare the joint's velocity change. I_eff = tau*dt/dv. Also compare the root's linear response
to a pure joint torque (coupling), as a check on the link inertia tensors."""
import os, sys, torch, numpy as np, mujoco
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_nexus.envs import nexusify
from isaaclab_nexus.physics.nexus_manager import NexusManager
TASK = "HeightTracking-G1-v0"; G1 = os.environ.get("NEXUS_G1_MJCF", "/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"); SRC = "/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
JOINTS = ["left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint", "waist_yaw_joint", "left_shoulder_pitch_joint", "left_elbow_joint"]; TAU = 20.0
cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); cfg.scene.num_envs = len(JOINTS); cfg.seed = 42; cfg.sim.gravity = (0.0, 0.0, 0.0)
for name in ("push_robot", "randomize_physics_material", "randomize_base_com"):
    if getattr(cfg.events, name, None) is not None: setattr(cfg.events, name, None)
nexusify(cfg, G1); env = gym.make(TASK, cfg=cfg).unwrapped; env.reset(); robot = env.scene.articulations["robot"]; dt = env.physics_dt
pose = robot.data.default_root_pose.torch.clone(); pose[:, 2] = 3.0; robot.write_root_pose_to_sim(pose)
q0 = robot.data.default_joint_pos.torch.clone(); robot.write_joint_state_to_sim(q0, torch.zeros_like(q0)); robot.write_root_velocity_to_sim(torch.zeros(len(JOINTS), 6, device=q0.device))
NexusManager.step(); robot.update(dt)                                     # settle one step with zero effort
v_before = robot.data.joint_vel.torch.clone(); rv_before = robot.data.root_lin_vel_w.torch.clone()
for e, jn in enumerate(JOINTS):                                          # env e: torque on joint jn only, bypassing the actuator
    j = robot.joint_names.index(jn); robot._effort[robot._cols[j], e] = TAU
NexusManager.step(); robot.update(dt); robot._effort[:, :] = 0.0
dv = robot.data.joint_vel.torch - v_before; drv = robot.data.root_lin_vel_w.torch - rv_before
# ---- MuJoCo reference from the ORIGINAL MJCF, same pose, zero-g, one step of the same dt
m = mujoco.MjModel.from_xml_path(SRC); m.opt.gravity[:] = 0; m.opt.timestep = dt; m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
def mj_ieff(jn):
    d = mujoco.MjData(m); jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn); dof = m.jnt_dofadr[jid]
    for k, name in enumerate(robot.joint_names):                            # same default pose
        jj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name); d.qpos[m.jnt_qposadr[jj]] = float(q0[0, k])
    d.qpos[2] = 3.0; mujoco.mj_forward(m, d); d.qfrc_applied[dof] = TAU; mujoco.mj_step(m, d)
    return TAU * dt / d.qvel[dof] if d.qvel[dof] != 0 else float('inf'), float(np.linalg.norm(d.qvel[:3]))
print(f"{'joint':<28}{'I_eff Nexus':>13}{'I_eff MuJoCo':>14}{'ratio':>8}   root |dv| Nexus / MuJoCo")
for e, jn in enumerate(JOINTS):
    j = robot.joint_names.index(jn); dvn = float(dv[e, j]); In = TAU * dt / dvn if dvn else float('inf'); Im, rmj = mj_ieff(jn)
    print(f"{jn:<28}{In:13.4f}{Im:14.4f}{In/Im:8.2f}   {float(drv[e].norm()):.4f} / {rmj:.4f}")
env.close(); app.close()
