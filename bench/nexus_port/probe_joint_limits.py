"""Are MJCF joint position limits enforced by the engine? Free-flying G1 (no gravity), zero PD gains: apply a
constant external torque on torso_link about the waist pitch axis (through the joint-space projection) and
watch waist_pitch: it must stop at the MJCF range, not keep turning."""
import os, torch, nexus3d
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_nexus import NexusCfg, NexusMjcfCfg
from isaaclab_nexus.physics.nexus_manager import NexusManager
G1 = "/workspace/bench/nexus_port/g1_29dof_convex64.xml"; N, DT = 2, 1 / 200
class _SimCfg: dt = DT; gravity = (0, 0, 0.0); device = "cuda:0"; physics = NexusCfg()
class _Sim: cfg = _SimCfg(); physics_manager = NexusManager
SimulationContext._instance = _Sim(); NexusManager.initialize(_Sim); NexusManager.ensure_envs(N)
robot = Articulation(ArticulationCfg(prim_path="/World/envs/env_.*/Robot", init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0)),
    spawn=NexusMjcfCfg(mjcf_path=G1, num_envs=N, auto_floor=False),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.5, effort_limit_sim=1000.0, velocity_limit_sim=1000.0)}))
robot.reset(); NexusManager.synchronize()
j = robot.joint_names.index("waist_pitch_joint"); tid = robot.find_bodies("torso_link")[0]
lim = robot.data.joint_pos_limits.torch[0, j].tolist(); print("waist_pitch MJCF limits as loaded:", [round(x, 3) for x in lim])
T = torch.zeros(N, 1, 3, device="cuda:0"); T[0, 0, 1] = 30.0; T[1, 0, 1] = -30.0                 # +/-30 N.m about y (waist pitch axis) on the torso
F = torch.zeros_like(T)
for i in range(120):
    robot.set_external_force_and_torque(F, T, body_ids=tid); robot.write_data_to_sim(); NexusManager.step(); robot.update(DT)
    if i in (9, 29, 59, 119): q = robot.data.joint_pos.torch[:, j]; v = robot.data.joint_vel.torch[:, j]; print(f"t={(i+1)*DT:.2f}s waist_pitch q {q.cpu().numpy().round(3)} v {v.cpu().numpy().round(1)} | projected tau {robot._wrench_tau[:, j].cpu().numpy().round(1)}")
