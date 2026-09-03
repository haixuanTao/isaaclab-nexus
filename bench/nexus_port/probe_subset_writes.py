"""Do write_joint_state_to_sim / write_root_pose_to_sim land on the right envs when env_ids is a SUBSET?"""
import os, torch, nexus3d
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_nexus import NexusCfg, NexusMjcfCfg
from isaaclab_nexus.physics.nexus_manager import NexusManager
G1 = "/workspace/bench/nexus_port/g1_29dof_convex64.xml"; N, DT = 64, 1 / 200
class _SimCfg: dt = DT; gravity = (0, 0, -9.81); device = "cuda:0"; physics = NexusCfg()
class _Sim: cfg = _SimCfg(); physics_manager = NexusManager
SimulationContext._instance = _Sim(); NexusManager.initialize(_Sim); NexusManager.ensure_envs(N)
st = NexusManager.state(); floor = nexus3d.ColliderBuilder.cuboid(20.0, 20.0, 0.5).build()
for e in range(N): st.insert_rigid_body_in(e, nexus3d.RigidBodyBuilder.fixed().translation(nexus3d.Vec3(0.0, 0.0, -0.5)).build(), floor)
robot = Articulation(ArticulationCfg(prim_path="/World/envs/env_.*/Robot", init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8)),
    spawn=NexusMjcfCfg(mjcf_path=G1, num_envs=N, auto_floor=False),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0, effort_limit=100.0)}))
robot.reset(); NexusManager.synchronize()
ids = torch.tensor([5, 17, 40, 63], device="cuda:0"); others = torch.tensor([0, 6, 41], device="cuda:0")
q = torch.zeros(len(ids), robot.num_joints, device="cuda:0"); q[:, :] = torch.arange(len(ids), device="cuda:0")[:, None] * 0.1 + 0.3   # env-specific values
robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
pose = torch.zeros(len(ids), 7, device="cuda:0"); pose[:, 2] = torch.tensor([1.0, 1.5, 2.0, 2.5], device="cuda:0"); pose[:, 3:7] = torch.tensor([0.7071, 0, 0, 0.7071], device="cuda:0")
robot.write_root_pose_to_sim(pose, env_ids=ids); NexusManager.synchronize(); robot.update(DT)
jp, rp, rq = robot.data.joint_pos.torch, robot.data.root_link_pos_w.torch, robot.data.root_link_quat_w.torch
print("after subset writes (no step):")
print("  target envs joint_pos[:,0]  :", jp[ids, 0].cpu().numpy().round(2), " expected [0.3 0.4 0.5 0.6]")
print("  other envs  joint_pos[:,0]  :", jp[others, 0].cpu().numpy().round(2), " expected default (~0 or default pose)")
print("  target envs root z          :", rp[ids, 2].cpu().numpy().round(2), " expected [1. 1.5 2. 2.5]")
print("  target envs root quat (xyzw):", rq[ids].cpu().numpy().round(3)[0], " expected [0.707 0 0 0.707]")
print("  other envs  root z          :", rp[others, 2].cpu().numpy().round(2), " expected 0.8")
NexusManager.step(); robot.update(DT); jp2 = robot.data.joint_pos.torch
print("after one physics step: target envs joint_pos[:,0]:", jp2[ids, 0].cpu().numpy().round(2), "| body_link z of foot (env 5):", robot.data.body_link_pos_w.torch[5, robot.find_bodies('left_ankle_roll_link')[0][0], 2].item().__round__(3))
