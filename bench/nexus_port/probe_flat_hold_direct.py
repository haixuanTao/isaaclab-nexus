"""Standing hold on a FLAT cuboid floor, direct Nexus scene (no manager env): G1 at its default
pose, PD hold toward the default pose at AGILE-like gains. Two floor frictions. If it stands here
but topples on the terrain tile, the collapse is a terrain-contact/spawn issue; if it topples here
too, the foot contact model."""
import os, sys, torch, nexus3d
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_nexus import NexusCfg, NexusMjcfCfg
from isaaclab_nexus.physics.nexus_manager import NexusManager
G1 = os.environ.get("NEXUS_G1_MJCF", "/workspace/bench/nexus_port/g1_29dof_convex64.xml"); FR = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
N, DT = 64, 1 / 200
class _SimCfg: dt = DT; gravity = (0, 0, -9.81); device = "cuda:0"; physics = NexusCfg()
class _Sim: cfg = _SimCfg(); physics_manager = NexusManager
SimulationContext._instance = _Sim(); NexusManager.initialize(_Sim); NexusManager.ensure_envs(N)
st = NexusManager.state()
floor = nexus3d.ColliderBuilder.cuboid(20.0, 20.0, 0.5).friction(FR).build()
for e in range(N): st.insert_rigid_body_in(e, nexus3d.RigidBodyBuilder.fixed().translation(nexus3d.Vec3(0.0, 0.0, -0.5)).build(), floor)
# AGILE's default standing pose (hips -0.1, knees 0.3, ankles -0.2) and leg/arm gains
init = ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.793), joint_pos={".*_hip_pitch_joint": -0.1, ".*_knee_joint": 0.3, ".*_ankle_pitch_joint": -0.2})
robot = Articulation(ArticulationCfg(prim_path="/World/envs/env_.*/Robot", init_state=init,
    spawn=NexusMjcfCfg(mjcf_path=G1, num_envs=N, auto_floor=False),
    actuators={"legs": ImplicitActuatorCfg(joint_names_expr=[".*_hip_.*", ".*_knee_joint"], stiffness=150.0, damping=5.0, effort_limit=139.0),
               "feet": ImplicitActuatorCfg(joint_names_expr=[".*_ankle_.*"], stiffness=40.0, damping=2.0, effort_limit=50.0),
               "rest": ImplicitActuatorCfg(joint_names_expr=["waist_.*", ".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*"], stiffness=40.0, damping=2.0, effort_limit=25.0)}))
robot.reset(); q0 = robot.data.default_joint_pos.torch.clone(); feet = robot.find_bodies(".*ankle_roll_link")[0]
NexusManager.synchronize(); fz0 = robot.data.body_link_pos_w.torch[:, feet, 2]; out = []
for i in range(600):
    robot.set_joint_position_target(q0); robot.write_data_to_sim(); NexusManager.step(); robot.update(DT)
    if i in (24, 99, 199, 399, 599):
        z = robot.data.root_link_pos_w.torch[:, 2]; out.append(f"t={(i+1)*DT:.2f}s z {float(z.mean()):.2f} up {float((z>0.6).float().mean())*100:.0f}%")
tq = robot.data.applied_torque.torch.abs()
print(f"[nexus FLAT floor friction {FR}] foot z at reset min/median {float(fz0.min()):.3f}/{float(fz0.median()):.3f} | " + " | ".join(out) + f" | |torque| median {float(tq.median()):.1f} max {float(tq.max()):.1f}")
