"""Does set_external_force_and_torque reach the physics on Nexus? Direct scene, G1 on a flat
floor, PD hold of the default pose. (a) +Z force of 500 N on torso_link (> 346 N weight): the robot
must rise. (b) pure +X torque on torso_link: the robot must roll about x. Reports root z / ang vel."""
import os, sys, torch, nexus3d
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_nexus import NexusCfg, NexusMjcfCfg
from isaaclab_nexus.physics.nexus_manager import NexusManager
G1 = "/workspace/bench/nexus_port/g1_29dof_convex64.xml"; N, DT = 16, 1 / 200
class _SimCfg: dt = DT; gravity = (0, 0, -9.81); device = "cuda:0"; physics = NexusCfg()
class _Sim: cfg = _SimCfg(); physics_manager = NexusManager
SimulationContext._instance = _Sim(); NexusManager.initialize(_Sim); NexusManager.ensure_envs(N)
st = NexusManager.state(); floor = nexus3d.ColliderBuilder.cuboid(20.0, 20.0, 0.5).friction(1.0).build()
for e in range(N): st.insert_rigid_body_in(e, nexus3d.RigidBodyBuilder.fixed().translation(nexus3d.Vec3(0.0, 0.0, -0.5)).build(), floor)
init = ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.793), joint_pos={".*_hip_pitch_joint": -0.1, ".*_knee_joint": 0.3, ".*_ankle_pitch_joint": -0.2})
robot = Articulation(ArticulationCfg(prim_path="/World/envs/env_.*/Robot", init_state=init, spawn=NexusMjcfCfg(mjcf_path=G1, num_envs=N, auto_floor=False),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=150.0, damping=5.0, effort_limit=139.0)}))
robot.reset(); q0 = robot.data.default_joint_pos.torch.clone(); torso = robot.find_bodies("torso_link")[0]
def run(label, F, T, steps=100):
    robot.reset(); robot.write_root_pose_to_sim(robot.data.default_root_pose.torch.clone()); robot.write_joint_state_to_sim(q0, torch.zeros_like(q0)); NexusManager.synchronize()
    f = torch.zeros(N, 1, 3, device="cuda:0"); t = torch.zeros(N, 1, 3, device="cuda:0"); f[:, 0] = torch.tensor(F, device="cuda:0"); t[:, 0] = torch.tensor(T, device="cuda:0")
    z0 = float(robot.data.root_link_pos_w.torch[:, 2].mean()); out = []
    for i in range(steps):
        robot.set_joint_position_target(q0); robot.set_external_force_and_torque(f, t, body_ids=torso); robot.write_data_to_sim(); NexusManager.step(); robot.update(DT)
        if i in (4, 9, 24, 49, 99): w = robot.data.root_ang_vel_w.torch.mean(0); out.append(f"t={(i+1)*DT:.2f}s z {float(robot.data.root_link_pos_w.torch[:,2].mean()):.3f} vz {float(robot.data.root_lin_vel_w.torch[:,2].mean()):+.2f} w=({float(w[0]):+.2f},{float(w[1]):+.2f},{float(w[2]):+.2f})")
    print(f"[{label}] z0 {z0:.3f} | " + " | ".join(out))
run("no external force   ", (0, 0, 0), (0, 0, 0))
run("+500 N up on torso  ", (0, 0, 500.0), (0, 0, 0))
run("+2000 N up on torso ", (0, 0, 2000.0), (0, 0, 0))
run("+100 Nm x-torque    ", (0, 0, 0), (100.0, 0, 0), steps=25)
run("+100 Nm y-torque    ", (0, 0, 0), (0, 100.0, 0), steps=25)
run("+100 Nm z-torque    ", (0, 0, 0), (0, 0, 100.0), steps=25)
