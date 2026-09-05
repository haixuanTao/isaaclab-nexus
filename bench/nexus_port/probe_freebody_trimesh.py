"""Minimal repro through the same path the robot uses (MJCF -> multibody with a free joint): one free body
whose colliders are VARIANT = box0 (2 cm box at origin) | boxoff (2 cm box at pos -0.03 z) | corners (the G1's
four 5 mm foot-corner geoms, verbatim) | hull (a G1 shin hull), dropped from 0.5 m on FLOOR=trimesh|cuboid.
Prints body z over time; the body should come to rest, not keep falling."""
import os, numpy as np, torch, nexus3d
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_nexus import NexusCfg, NexusMjcfCfg
from isaaclab_nexus.physics.nexus_manager import NexusManager
FLOOR, VAR = os.environ.get("FLOOR", "trimesh"), os.environ.get("VARIANT", "box0")
GEOMS = {"box0": '<geom type="box" size="0.02 0.02 0.02"/>',
         "boxoff": '<geom type="box" size="0.02 0.02 0.02" pos="0.05 0.02 -0.03"/>',
         "corners": '<geom size="0.005" pos="-0.05 0.025 -0.03"/><geom size="0.005" pos="-0.05 -0.025 -0.03"/><geom size="0.005" pos="0.12 0.03 -0.03"/><geom size="0.005" pos="0.12 -0.03 -0.03"/>',
         "hull": '<geom type="mesh" mesh="shin"/>'}[VAR]
mj = f'''<mujoco model="drop"><compiler angle="radian"/><asset><mesh name="shin" file="/workspace/bench/nexus_port/meshes_convex64/left_ankle_pitch_link.STL"/></asset>
<worldbody><body name="foot" pos="0 0 0.5"><freejoint/><inertial pos="0 0 0" mass="1" diaginertia="0.001 0.001 0.001"/>{GEOMS}<body name="flag" pos="0 0 0.3"><joint name="hinge" type="hinge" axis="0 1 0"/><inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/><geom type="box" size="0.01 0.01 0.01" contype="0" conaffinity="0"/></body></body></worldbody></mujoco>'''
path = f"/tmp/claude-0/-workspace/c6ac0505-64b1-4feb-84e0-f8ea7a9c8078/scratchpad/drop_{VAR}.xml"; open(path, "w").write(mj)
N, DT = 4, 1 / 200
class _SimCfg: dt = DT; gravity = (0, 0, -9.81); device = "cuda:0"; physics = NexusCfg()
class _Sim: cfg = _SimCfg(); physics_manager = NexusManager
SimulationContext._instance = _Sim(); NexusManager.initialize(_Sim); NexusManager.ensure_envs(N)
st = NexusManager.state()
if FLOOR == "cuboid":
    col = nexus3d.ColliderBuilder.cuboid(4.0, 4.0, 0.5).build(); tz = -0.5
else:
    xs = np.arange(-4.0, 4.0 + 1e-6, 0.25); nx = len(xs); X, Y = np.meshgrid(xs, xs, indexing="ij")
    V = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1); a = (np.arange(nx - 1)[:, None] * nx + np.arange(nx - 1)[None, :]).ravel()
    F = np.concatenate([np.stack([a, a + 1, a + nx + 1], 1), np.stack([a, a + nx + 1, a + nx], 1)], 0)[:, [0, 2, 1]]
    col = nexus3d.ColliderBuilder.trimesh([tuple(map(float, v)) for v in V], [tuple(map(int, f)) for f in F]).build(); tz = 0.0
for e in range(N): st.insert_rigid_body_in(e, nexus3d.RigidBodyBuilder.fixed().translation(nexus3d.Vec3(0.0, 0.0, tz)).build(), col)
robot = Articulation(ArticulationCfg(prim_path="/World/envs/env_.*/Robot", init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, float(os.environ.get("START_Z", "0.5")))),
                                     spawn=NexusMjcfCfg(mjcf_path=path, num_envs=N, auto_floor=False), actuators={"j": ImplicitActuatorCfg(joint_names_expr=["hinge"], stiffness=0.0, damping=0.01, effort_limit=1.0)}))
robot.reset(); NexusManager.synchronize(); zs = []
for i in range(400):
    robot.write_data_to_sim(); NexusManager.step(); robot.update(DT)
    if (i + 1) in (50, 100, 200, 400): zs.append(f"t={(i+1)*DT:.2f}s z={float(robot.data.root_link_pos_w.torch[0, 2]):+.3f}")
print(f"[{FLOOR:7s} | {VAR:7s}] " + " ".join(zs))
