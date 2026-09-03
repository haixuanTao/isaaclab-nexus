"""(1) Root yaw after construction / reset / an explicit identity pose write. (2) Does a world root
velocity write come back as written (frame of the free-joint velocity DOFs)?"""
import os, torch, nexus3d
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_nexus import NexusCfg, NexusMjcfCfg
from isaaclab_nexus.physics.nexus_manager import NexusManager
G1 = "/workspace/bench/nexus_port/g1_29dof_convex64.xml"; N, DT = 4, 1 / 200
class _SimCfg: dt = DT; gravity = (0, 0, 0); device = "cuda:0"; physics = NexusCfg()
class _Sim: cfg = _SimCfg(); physics_manager = NexusManager
SimulationContext._instance = _Sim(); NexusManager.initialize(_Sim)
robot = Articulation(ArticulationCfg(prim_path="/World/envs/env_.*/Robot", init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 3.0)),
    spawn=NexusMjcfCfg(mjcf_path=G1, num_envs=N, auto_floor=False), actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=150.0, damping=5.0, effort_limit=139.0)}))
NexusManager.synchronize(); lay = robot._lay; q = lambda: [round(x, 3) for x in robot.data.root_link_quat_w.torch[0].tolist()]
raw = lambda: [round(x, 3) for x in robot._ws[robot._root_link, lay["WS_JOINT_ROT"], 0, :].tolist()]
print("after construction : root quat wxyz", q(), "| raw JOINT_ROT xyzw", raw(), "| default_root_pose rot", [round(x,3) for x in robot.data.default_root_pose.torch[0,3:].tolist()])
robot.reset(); NexusManager.synchronize(); print("after reset()      : root quat wxyz", q(), "| raw JOINT_ROT xyzw", raw())
pose = robot.data.default_root_pose.torch.clone(); pose[:, 3:] = torch.tensor([1.0, 0, 0, 0], device="cuda:0"); robot.write_root_pose_to_sim(pose); NexusManager.step(); robot.update(DT)
print("after identity write + 1 step: root quat wxyz", q(), "| raw JOINT_ROT xyzw", raw())
hip = robot.find_bodies("left_hip_pitch_link")[0][0]; d = robot.data.body_link_pos_w.torch[0, hip] - robot.data.root_link_pos_w.torch[0]
print("left hip position relative to pelvis (world):", [round(x, 3) for x in d.tolist()], "| MJCF: left hip is at +y (0.064)")
v = torch.zeros(N, 6, device="cuda:0"); v[:, 0] = 1.0; robot.write_root_velocity_to_sim(v); NexusManager.step(); robot.update(DT)
print("wrote root lin vel (+1,0,0) world; read back after 1 step:", [round(x, 3) for x in robot.data.root_lin_vel_w.torch[0].tolist()])
