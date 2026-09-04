"""Does the per-substep PD hook run? Bare articulation: set a target, write_data_to_sim once, step 3 times WITHOUT
rewriting, and watch applied_torque change (it stays frozen without the hook)."""
import os, torch, nexus3d
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_nexus import NexusCfg, NexusMjcfCfg
from isaaclab_nexus.physics.nexus_manager import NexusManager
G1 = "/workspace/bench/nexus_port/g1_29dof_convex64.xml"; N, DT = 2, 1 / 200
class _SimCfg: dt = DT; gravity = (0, 0, -9.81); device = "cuda:0"; physics = NexusCfg()
class _Sim: cfg = _SimCfg(); physics_manager = NexusManager
SimulationContext._instance = _Sim(); NexusManager.initialize(_Sim); NexusManager.ensure_envs(N)
robot = Articulation(ArticulationCfg(prim_path="/World/envs/env_.*/Robot", init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0)),
    spawn=NexusMjcfCfg(mjcf_path=G1, num_envs=N, auto_floor=False),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=50.0, damping=1.0, effort_limit_sim=1000.0, velocity_limit_sim=100.0)}))
robot.reset(); NexusManager.synchronize(); print("hooks registered:", [h.__name__ for h in NexusManager.post_step_hooks])
j = robot.find_joints("left_elbow_joint")[0][0]; q0 = robot.data.default_joint_pos.torch.clone(); tgt = q0.clone(); tgt[:, j] += 1.0
robot.set_joint_position_target(tgt); robot.write_data_to_sim(); NexusManager.synchronize()
for i in range(4):
    NexusManager.step(); NexusManager.synchronize(); robot.update(DT)
    print(f"after physics step {i+1}: elbow q-q0 {float(robot.data.joint_pos.torch[0, j]-q0[0, j]):+.4f} v {float(robot.data.joint_vel.torch[0, j]):+.2f} | applied torque {float(robot.data.applied_torque.torch[0, j]):+.2f} | engine effort row {float(robot._effort[robot._cols[j], 0]):+.2f}")
