"""Factory-dispatch proof for the Nexus Isaac Lab backend, without Kit.

`SimulationContext` normally requires an Isaac Sim app. Its singleton is a
plain class attribute, so a test-only shim stands in for it here; everything
else is the real Isaac Lab machinery:

  isaaclab.assets.Articulation(cfg)          # the real factory class
    -> FactoryBase.__new__
    -> _get_backend()  -> "nexus"            # via the patched backend_utils
    -> import isaaclab_nexus.assets.articulation, getattr(..., "Articulation")
    -> Nexus CUDA state, MJCF humanoid x N envs, zero-copy torch views.
"""
import math
import torch

import isaaclab.sim as _sim
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg

from isaaclab_nexus import NexusCfg, NexusMjcfCfg
from isaaclab_nexus.physics.nexus_manager import NexusManager

MJCF = "/workspace/WBC-AGILE/.venv/lib/python3.12/site-packages/newton/examples/assets/nv_humanoid.xml"
NENV = 8


class _SimCfg:            # the fields NexusManager/finalize read from sim.cfg
    dt = 1.0 / 200.0
    gravity = (0.0, 0.0, -9.81)
    device = "cuda:0"
    physics = NexusCfg()


class _SimShim:           # stands in for SimulationContext (no Kit)
    cfg = _SimCfg()
    physics_manager = NexusManager


assert SimulationContext.instance() is None, "a real SimulationContext exists; this test expects none"
SimulationContext._instance = _SimShim()
NexusManager.initialize(_SimShim)          # base sets _sim/_cfg/_device; then Nexus backend+state
assert NexusManager.is_cuda(), "NexusManager did not come up on CUDA"
print("physics manager:", NexusManager.__name__, "| backend cuda:", NexusManager.is_cuda())

cfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=NexusMjcfCfg(mjcf_path=MJCF, num_envs=NENV),
    actuators={},
)
robot = Articulation(cfg)                  # <- real Isaac Lab factory dispatch
print("factory resolved to:", type(robot).__module__ + "." + type(robot).__name__)
assert type(robot).__module__.startswith("isaaclab_nexus."), "factory did not dispatch to the Nexus backend"

print(f"num_instances={robot.num_instances} num_bodies={robot.num_bodies} num_joints={robot.num_joints}")
assert robot.num_instances == NENV

d = robot.data
q = d.joint_coords
p = d.body_link_pose_w
v = d.joint_vel
print("joint_coords     ", tuple(q.shape), q.device, "| view of nexus mem:", q.data_ptr() >= robot._ws_view.ptr)
print("body_link_pose_w ", tuple(p.shape))
print("joint_vel (flat) ", tuple(v.shape), "| zero-copy:", v.data_ptr() >= robot._dof_view.ptr)
print("root_link_pos_w  ", tuple(d.root_link_pos_w.shape), "env0:", [round(x, 3) for x in d.root_link_pos_w[0].tolist()])

# step through the manager exactly as SimulationContext.step() would
z0 = d.root_link_pos_w[:, 2].clone()
for _ in range(100):
    NexusManager.step()
robot.update(_SimCfg.dt)
z1 = d.root_link_pos_w[:, 2]
dz = (z1 - z0)
print(f"root z after 100 steps ({100*_SimCfg.dt:.2f}s): env0 {z0[0]:.3f} -> {z1[0]:.3f}  (all envs move: {(dz.abs() > 1e-4).all().item()})")
assert (dz.abs() > 1e-4).all(), "robot did not move under gravity in every env"
# free fall bound: |dz| <= 0.5 g t^2 (contacts/joints can only slow it)
t = 100 * _SimCfg.dt
assert dz.abs().max().item() <= 0.5 * 9.81 * t * t + 0.05, "moved more than free fall allows"

# the unimplemented surface must fail loudly, not silently
try:
    robot.write_joint_position_to_sim_index(None, None)
    raise SystemExit("expected NotImplementedError")
except NotImplementedError as e:
    print("unimplemented call fails loudly:", str(e)[:70])

print("ISAAC LAB FACTORY -> NEXUS CUDA BACKEND: OK")
