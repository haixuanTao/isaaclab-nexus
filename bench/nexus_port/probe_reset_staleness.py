"""Are the articulation views fresh after a reset + forward(), before any physics step?"""
import os, torch
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_nexus import NexusCfg, NexusMjcfCfg
from isaaclab_nexus.physics.nexus_manager import NexusManager
G1 = "/workspace/bench/nexus_port/g1_29dof_convex64.xml"; N, DT = 8, 1 / 200
class _SimCfg: dt = DT; gravity = (0, 0, -9.81); device = "cuda:0"; physics = NexusCfg()
class _Sim: cfg = _SimCfg(); physics_manager = NexusManager
SimulationContext._instance = _Sim(); NexusManager.initialize(_Sim)
robot = Articulation(ArticulationCfg(prim_path="/World/envs/env_.*/Robot", spawn=NexusMjcfCfg(mjcf_path=G1, num_envs=N, auto_floor=True),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=50.0, damping=2.0, effort_limit=100.0)}))
for _ in range(100): robot.write_data_to_sim(); NexusManager.step(); robot.update(DT)      # let it fall over (terminal-like pose)
old_root = robot.data.root_link_pos_w.torch.clone(); old_body = robot.data.body_link_pos_w.torch.clone(); old_q = robot.data.joint_pos.torch.clone()
print(f"before reset: root z mean {float(old_root[:,2].mean()):.3f}")
robot.reset()                                                          # template reset (upright default)
pose = robot.data.default_root_pose.torch.clone(); robot.write_root_pose_to_sim(pose); robot.write_joint_state_to_sim(robot.data.default_joint_pos.torch.clone(), torch.zeros_like(old_q))
NexusManager.forward(); NexusManager.synchronize()
r, b, q = robot.data.root_link_pos_w.torch, robot.data.body_link_pos_w.torch, robot.data.joint_pos.torch
print(f"after reset + forward(), BEFORE any step: root z mean {float(r[:,2].mean()):.3f} (written {float(pose[0,2]):.3f}) | root pos stale? {torch.allclose(r, old_root)} | body poses stale? {torch.allclose(b, old_body)} | joint_pos == default? {torch.allclose(q, robot.data.default_joint_pos.torch, atol=1e-4)}")
robot.update(DT); r2 = robot.data.root_link_pos_w.torch; print(f"after update() (no step): root z mean {float(r2[:,2].mean()):.3f} | stale? {torch.allclose(r2, old_root)}")
robot.write_data_to_sim(); NexusManager.step(); robot.update(DT); r3 = robot.data.root_link_pos_w.torch; print(f"after one physics step: root z mean {float(r3[:,2].mean()):.3f}")
