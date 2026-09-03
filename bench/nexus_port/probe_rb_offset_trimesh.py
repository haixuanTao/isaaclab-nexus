"""Minimal repro: ONE dynamic rigid body (2 cm cube or 2 cm ball), with or without a collider-local offset,
dropped from 0.5 m on a flat trimesh vs a cuboid floor. Prints the body z over time (rests at ~0.02 if the
collider is centered, at ~0.05 if the collider hangs 3 cm below the body origin)."""
import os, numpy as np, nexus3d
from isaaclab.sim import SimulationContext
from isaaclab_nexus import NexusCfg
from isaaclab_nexus.physics.nexus_manager import NexusManager
FLOOR, OFFSET, SHAPE = os.environ.get("FLOOR", "trimesh"), os.environ.get("OFFSET", "0") == "1", os.environ.get("SHAPE", "cuboid")
DT = 1 / 200
class _SimCfg: dt = DT; gravity = (0, 0, -9.81); device = "cuda:0"; physics = NexusCfg()
class _Sim: cfg = _SimCfg(); physics_manager = NexusManager
SimulationContext._instance = _Sim(); NexusManager.initialize(_Sim); NexusManager.ensure_envs(1)
st, be = NexusManager.state(), NexusManager._backend
if FLOOR == "cuboid":
    col = nexus3d.ColliderBuilder.cuboid(4.0, 4.0, 0.5).build(); tz = -0.5
else:
    xs = np.arange(-4.0, 4.0 + 1e-6, 0.25); nx = len(xs); X, Y = np.meshgrid(xs, xs, indexing="ij")
    V = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1); a = (np.arange(nx - 1)[:, None] * nx + np.arange(nx - 1)[None, :]).ravel()
    F = np.concatenate([np.stack([a, a + 1, a + nx + 1], 1), np.stack([a, a + nx + 1, a + nx], 1)], 0)[:, [0, 2, 1]]
    col = nexus3d.ColliderBuilder.trimesh([tuple(map(float, v)) for v in V], [tuple(map(int, f)) for f in F]).build(); tz = 0.0
st.insert_rigid_body_in(0, nexus3d.RigidBodyBuilder.fixed().translation(nexus3d.Vec3(0.0, 0.0, tz)).build(), col)
cb = nexus3d.ColliderBuilder.cuboid(0.02, 0.02, 0.02) if SHAPE == "cuboid" else nexus3d.ColliderBuilder.ball(0.02)
if OFFSET: cb = cb.translation(nexus3d.Vec3(0.05, 0.02, -0.03))
h = st.insert_rigid_body_in(0, nexus3d.RigidBodyBuilder.dynamic().translation(nexus3d.Vec3(0.0, 0.0, 0.5)).build(), cb.build())
if hasattr(NexusManager, "_finalize") and not NexusManager._finalized: NexusManager._finalize()
zs = []
for i in range(400):
    NexusManager.step()
    if (i + 1) in (50, 100, 200, 400): zs.append(f"t={(i+1)*DT:.2f}s z={float(st.body_poses(be)[1, 2]):+.3f}")
print(f"[{FLOOR} | {SHAPE} r/half=0.02 | collider offset {'(0.05,0.02,-0.03)' if OFFSET else 'none'}] " + " ".join(zs) + f"  (expected rest z {0.05 if OFFSET else 0.02:.2f})")
