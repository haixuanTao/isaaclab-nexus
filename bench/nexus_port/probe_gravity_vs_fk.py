"""Are Isaac's body-frame observations consistent with the engine's kinematics now? Write a pitched
root orientation in (x, y, z, w), then compare `projected_gravity` computed the Isaac way
(quat_apply_inverse(root_quat_w, -z)) with the gravity direction implied by FORWARD KINEMATICS
(the torso-relative-to-pelvis vector, which the engine rotates with its own quaternion)."""
import os, math, torch
import isaaclab.utils.math as mu
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
NexusManager.synchronize(); q0 = robot.data.default_joint_pos.torch.clone(); torso = robot.find_bodies("torso_link")[0][0]; hip = robot.find_bodies("left_hip_pitch_link")[0][0]
print("root quat (x,y,z,w) after construction:", [round(x, 3) for x in robot.data.root_link_quat_w.torch[0].tolist()], "| left hip rel pelvis (world):", [round(x, 3) for x in (robot.data.body_link_pos_w.torch[0, hip] - robot.data.root_link_pos_w.torch[0]).tolist()], "(MJCF: +0.064 in y)")
up0 = (robot.data.body_link_pos_w.torch[0, torso] - robot.data.root_link_pos_w.torch[0]); up0 = up0 / up0.norm()      # torso direction in the pelvis frame at identity
for name, axis, ang in (("pitch +30deg about y", (0.0, 1.0, 0.0), math.radians(30)), ("roll -45deg about x", (1.0, 0.0, 0.0), math.radians(-45)), ("yaw +90deg about z", (0.0, 0.0, 1.0), math.radians(90))):
    s, c = math.sin(ang / 2), math.cos(ang / 2); qx = torch.tensor([[axis[0] * s, axis[1] * s, axis[2] * s, c]], device="cuda:0").expand(N, 4)   # (x, y, z, w)
    pose = robot.data.default_root_pose.torch.clone(); pose[:, 3:] = qx; robot.write_root_pose_to_sim(pose); robot.write_joint_state_to_sim(q0, torch.zeros_like(q0)); NexusManager.step(); robot.update(DT)
    qr = robot.data.root_link_quat_w.torch[0:1]
    g_isaac = mu.quat_apply_inverse(qr, torch.tensor([[0.0, 0.0, -1.0]], device="cuda:0"))[0]            # what AGILE's projected_gravity computes
    g_fk = mu.quat_apply_inverse(qx[0:1], torch.tensor([[0.0, 0.0, -1.0]], device="cuda:0"))[0]           # expected from the orientation we wrote
    up = robot.data.body_link_pos_w.torch[0, torso] - robot.data.root_link_pos_w.torch[0]; up = up / up.norm()
    up_expect = mu.quat_apply(qx[0:1], up0[None])[0]                                                           # engine FK must rotate the torso direction the same way
    print(f"{name:<22} read-back quat {[round(x,3) for x in qr[0].tolist()]} | projected_gravity Isaac {[round(x,3) for x in g_isaac.tolist()]} expected {[round(x,3) for x in g_fk.tolist()]} | torso dir FK {[round(x,3) for x in up.tolist()]} expected {[round(x,3) for x in up_expect.tolist()]}")
