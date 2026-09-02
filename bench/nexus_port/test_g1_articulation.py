"""G1 29-DOF through the Nexus Articulation with AGILE's own cfg (DelayedDCMotor actuators):
stand at the default pose, state-write readbacks, body wrench on the torso."""
import math, torch
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import math as mu
from isaaclab_nexus import NexusCfg, NexusMjcfCfg
from isaaclab_nexus.physics.nexus_manager import NexusManager
from agile.rl_env.assets.robots import unitree_g1
G1 = "/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"; NENV, DT = 4, 1 / 200
class _SimCfg: dt = DT; gravity = (0, 0, -9.81); device = "cuda:0"; physics = NexusCfg()
class _Sim: cfg = _SimCfg(); physics_manager = NexusManager
SimulationContext._instance = _Sim(); NexusManager.initialize(_Sim)
cfg = unitree_g1.G1_29DOF_HEIGHT_TRACKING.replace(prim_path="/World/envs/env_.*/Robot", spawn=NexusMjcfCfg(mjcf_path=G1, num_envs=NENV, auto_floor=True))
robot = Articulation(cfg); d = robot.data
print(f"{type(robot).__module__} | joints {robot.num_joints} bodies {robot.num_bodies} | actuators {list(robot.actuators)} -> {[type(a).__name__ for a in robot.actuators.values()]}")
print("default joint pos (nonzero):", {n: round(float(v), 3) for n, v in zip(robot.joint_names, d.default_joint_pos.torch[0]) if abs(v) > 1e-6})
print("init root pose:", [round(v, 3) for v in d.root_pose_w.torch[0].tolist()], "| limits sample", d.joint_pos_limits.torch[0, :2].tolist(), "| mass total %.2f" % d.body_mass.torch[0].sum())
print("effort limits:", d.joint_effort_limits.torch[0, [0, 3, 12, 15]].tolist(), "| vel limits:", d.joint_vel_limits.torch[0, [0, 3, 12, 15]].tolist())
feet = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", history_length=3, track_air_time=True))
print("contact sensor bodies:", feet.num_sensors)
robot.set_joint_position_target(d.default_joint_pos.torch.clone())
for i in range(600):
    robot.write_data_to_sim(); NexusManager.step(); robot.update(DT); feet.update(DT)
err = (d.joint_pos.torch - d.default_joint_pos.torch).abs()
print(f"after 3 s: root z {d.root_pos_w.torch[:, 2].tolist()} | joint err mean {err.mean():.3f} max {err.max():.3f} rad | |applied torque| max {d.applied_torque.torch.abs().max():.1f} Nm | proj gravity {[round(v,3) for v in d.projected_gravity_b.torch[0].tolist()]}")
print("feet forces z:", [round(v, 1) for v in feet.data.net_forces_w.torch[0, robot.find_bodies('.*ankle_roll_link')[0], 2].tolist()], "| total sensed / weight:", round(feet.data.net_forces_w.torch[0, :, 2].sum().item() / (d.body_mass.torch[0].sum().item() * 9.81), 3))
assert (d.root_pos_w.torch[:, 2] > 0.6).all() and err.mean() < 0.15, "G1 does not stand at its default pose"
# ---- state writes: yaw 90 deg, lift to 1.0 m, joints = default, zero vel; readback after one step
q = mu.quat_from_euler_xyz(torch.zeros(NENV, device="cuda"), torch.zeros(NENV, device="cuda"), torch.full((NENV,), math.pi / 2, device="cuda"))
pose = torch.cat([torch.tensor([[0.3, -0.2, 1.0]], device="cuda").expand(NENV, 3), q], 1)
robot.write_root_pose_to_sim(pose); robot.write_root_velocity_to_sim(torch.zeros(NENV, 6, device="cuda"))
robot.write_joint_state_to_sim(d.default_joint_pos.torch, torch.zeros_like(d.default_joint_pos.torch))
robot.write_data_to_sim(); NexusManager.step(); robot.update(DT)
rp = d.root_pose_w.torch[0]; print("readback root pose:", [round(v, 3) for v in rp.tolist()], "| heading", round(float(d.heading_w.torch[0]), 3), "| joint err", round(float((d.joint_pos.torch - d.default_joint_pos.torch).abs().max()), 4))
assert (rp[:3] - pose[0, :3]).abs().max() < 0.06 and abs(float(d.heading_w.torch[0]) - math.pi / 2) < 0.02
# ---- body wrench on the torso: 1.5 x weight upward for 0.25 s -> root accelerates up
tid = robot.find_bodies("torso_link")[0]; W = d.body_mass.torch[0].sum().item() * 9.81
F = torch.zeros(NENV, len(tid), 3, device="cuda"); F[..., 2] = 1.5 * W
z0 = d.root_pos_w.torch[:, 2].clone(); v0 = d.root_lin_vel_w.torch[:, 2].clone()
for i in range(50):
    robot.permanent_wrench_composer.set_forces_and_torques(forces=F, torques=torch.zeros_like(F), body_ids=tid, is_global=True)
    robot.write_data_to_sim(); NexusManager.step(); robot.update(DT)
print(f"wrench: root vz {v0[0]:.2f} -> {d.root_lin_vel_w.torch[0, 2]:.2f} m/s, z {z0[0]:.3f} -> {d.root_pos_w.torch[0, 2]:.3f}")
assert d.root_lin_vel_w.torch[0, 2] > v0[0] + 0.5
# ---- reset restores the template
robot.reset(torch.tensor([0, 2], device="cuda")); robot.write_data_to_sim(); NexusManager.step(); robot.update(DT)
print("after reset(0,2): root z", [round(v, 3) for v in d.root_pos_w.torch[:, 2].tolist()])
print("G1 ARTICULATION OK")
