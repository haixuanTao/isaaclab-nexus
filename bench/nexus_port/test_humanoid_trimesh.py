"""Trimesh contact for MULTIBODY links: drop the humanoid onto a raised trimesh quad vs a raised cuboid.
Feet resting at the slab height (0.5) => trimesh contacts work; feet at the loader floor (~ -0.1) => they don't."""
import torch, nexus3d
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_nexus import NexusCfg, NexusMjcfCfg
from isaaclab_nexus.physics.nexus_manager import NexusManager
MJCF = "/workspace/WBC-AGILE/.venv/lib/python3.12/site-packages/newton/examples/assets/nv_humanoid.xml"
NENV, DT, ZS = 2, 1/200, 0.5
class _SimCfg: dt = DT; gravity = (0, 0, -9.81); device = "cuda:0"; physics = NexusCfg()
class _Sim: cfg = _SimCfg(); physics_manager = NexusManager
SimulationContext._instance = _Sim(); NexusManager.initialize(_Sim)
NexusManager.ensure_envs(NENV); st = NexusManager.state()
v = [(-3, -3, ZS), (3, -3, ZS), (3, 3, ZS), (-3, 3, ZS)]
st.insert_rigid_body_in(0, nexus3d.RigidBodyBuilder.fixed().build(), nexus3d.ColliderBuilder.trimesh(v, [(0, 1, 2), (0, 2, 3)]).build())        # env0: trimesh
st.insert_rigid_body_in(1, nexus3d.RigidBodyBuilder.fixed().translation(nexus3d.Vec3(0, 0, ZS - 0.05)).build(), nexus3d.ColliderBuilder.cuboid(3, 3, 0.05).build())  # env1: cuboid
robot = Articulation(ArticulationCfg(prim_path="/World/envs/env_.*/Robot", spawn=NexusMjcfCfg(mjcf_path=MJCF, num_envs=NENV),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=80.0, damping=5.0, effort_limit=200.0)}))
d = robot.data; be = NexusManager.backend()
st.reset_envs(be, [0, 1], [[0, 0, ZS + 0.6]] * 2, [0.0] * (2 * robot._lay["dofs_per_batch"]))     # start 0.6 m above the slabs
robot.set_joint_position_target(torch.zeros(NENV, robot.num_joints, device="cuda"))
for _ in range(600): robot.write_data_to_sim(); NexusManager.step()
robot.update(DT)
feet, _ = robot.find_bodies(".*foot")
fz = d.body_link_pos_w.torch[:, feet, 2].min(dim=1).values
print(f"lowest foot z: env0 (trimesh) = {fz[0]:.3f}   env1 (cuboid) = {fz[1]:.3f}   slab top = {ZS}")
print("trimesh contact:", "WORKS" if fz[0] > ZS - 0.15 else "FAILS (fell through)", "| cuboid contact:", "WORKS" if fz[1] > ZS - 0.15 else "FAILS")
