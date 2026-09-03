"""Do bodies fall THROUGH a flat terrain trimesh? Direct Nexus scene (no manager env, no kit): G1 spawned
in 4 orientations (upright / on back / on front / on side) 0.5 m above a floor at z=0, PD toward the default
pose. FLOOR=cuboid|trimesh (trimesh = the same 8x8 m, RES-m grid of triangle pairs terrain.py builds).
Reports the lowest body z over envs; anything well below 0 is penetration, below -0.4 is fall-through."""
import os, sys, numpy as np, torch, nexus3d
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_nexus import NexusCfg, NexusMjcfCfg
from isaaclab_nexus.physics.nexus_manager import NexusManager
G1 = os.environ.get("NEXUS_G1_MJCF", "/workspace/bench/nexus_port/g1_29dof_convex64.xml")
FLOOR = os.environ.get("FLOOR", "trimesh"); RES = float(os.environ.get("RES", "0.25")); SUB = int(os.environ.get("SUBSTEPS", "1")); CAP = int(os.environ.get("CAP", "256"))
N, DT = 64, 1 / 200
class _SimCfg: dt = DT; gravity = (0, 0, -9.81); device = "cuda:0"; physics = NexusCfg(solver_iterations=SUB, collisions_capacity=CAP, contact_reduction=os.environ.get('REDUCE', '1') == '1')
class _Sim: cfg = _SimCfg(); physics_manager = NexusManager
SimulationContext._instance = _Sim(); NexusManager.initialize(_Sim); NexusManager.ensure_envs(N)
st = NexusManager.state()
if FLOOR == "cuboid":
    col = nexus3d.ColliderBuilder.cuboid(4.0, 4.0, 0.5).friction(1.0).build(); tz = -0.5
else:
    xs = np.arange(-4.0, 4.0 + 1e-6, RES); nx = len(xs); X, Y = np.meshgrid(xs, xs, indexing="ij")
    V = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1); a = (np.arange(nx - 1)[:, None] * nx + np.arange(nx - 1)[None, :]).ravel()
    F = np.concatenate([np.stack([a, a + 1, a + nx + 1], 1), np.stack([a, a + nx + 1, a + nx], 1)], 0)
    if os.environ.get('FLIP', '0') == '1': F = F[:, [0, 2, 1]]                                        # normals up
    col = nexus3d.ColliderBuilder.trimesh([tuple(map(float, v)) for v in V], [tuple(map(int, f)) for f in F]).friction(1.0).build(); tz = 0.0
for e in range(N): st.insert_rigid_body_in(e, nexus3d.RigidBodyBuilder.fixed().translation(nexus3d.Vec3(0.0, 0.0, tz)).build(), col)
backstop = nexus3d.ColliderBuilder.cuboid(4.0, 4.0, 0.5).build()                          # like terrain.py: 0.5 m below
for e in range(N): st.insert_rigid_body_in(e, nexus3d.RigidBodyBuilder.fixed().translation(nexus3d.Vec3(0.0, 0.0, -1.0)).build(), backstop)
init = ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5), joint_pos={".*_hip_pitch_joint": -0.1, ".*_knee_joint": 0.3, ".*_ankle_pitch_joint": -0.2})
robot = Articulation(ArticulationCfg(prim_path="/World/envs/env_.*/Robot", init_state=init,
    spawn=NexusMjcfCfg(mjcf_path=G1, num_envs=N, auto_floor=False),
    actuators={"legs": ImplicitActuatorCfg(joint_names_expr=[".*_hip_.*", ".*_knee_joint"], stiffness=150.0, damping=5.0, effort_limit=139.0),
               "feet": ImplicitActuatorCfg(joint_names_expr=[".*_ankle_.*"], stiffness=40.0, damping=2.0, effort_limit=50.0),
               "rest": ImplicitActuatorCfg(joint_names_expr=["waist_.*", ".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*"], stiffness=40.0, damping=2.0, effort_limit=25.0)}))
robot.reset()
s = 0.70710678; quats = torch.tensor([[0, 0, 0, 1.0], [s, 0, 0, s], [-s, 0, 0, s], [0, s, 0, s]], device="cuda:0")   # xyzw: upright, back, front, side
pose = robot.data.default_root_state.torch.clone(); pose[:, :3] = torch.tensor([0, 0, float(os.environ.get("DROP_Z", "0.5"))], device="cuda:0"); pose[:, 3:7] = quats[torch.arange(N, device="cuda:0") % 4]
robot.write_root_pose_to_sim(pose[:, :7]); robot.write_root_velocity_to_sim(torch.zeros(N, 6, device="cuda:0"))
q0 = robot.data.default_joint_pos.torch.clone(); robot.write_joint_state_to_sim(q0, torch.zeros_like(q0)); NexusManager.synchronize()
low = []
for i in range(800):
    robot.set_joint_position_target(q0); robot.write_data_to_sim(); NexusManager.step(); robot.update(DT)
    low.append(robot.data.body_link_pos_w.torch[..., 2].min(1).values.cpu().numpy())
low = np.stack(low); out = []
for t in (0.25, 0.5, 1, 2, 4):
    v = low[min(int(t / DT) - 1, 799)]; out.append(f"t={t:.2f}s lowest-body z median {np.median(v):+.3f} min {v.min():+.3f} through(<-0.4) {(v < -0.4).mean()*100:.0f}%")
by_pose = [f"{n}: min {low[:, k::4].min():+.2f}" for k, n in enumerate(("upright", "back", "front", "side"))]
print(f"[{FLOOR} floor res {RES} substeps {SUB} cap {CAP}] " + " | ".join(out) + " || per pose " + ", ".join(by_pose))
