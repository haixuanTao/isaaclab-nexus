"""A/B: per-collider-pair contact reduction on the AGILE rough-terrain tile.

Same scene twice (fresh process per run, arg 0|1): humanoid dropped on a real
AGILE terrain tile, settle, then report foot penetration and contact counts.
usage: test_contact_reduction.py <0|1>
"""
import sys, time, torch
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab_nexus import NexusCfg, NexusMjcfCfg
from isaaclab_nexus.physics.nexus_manager import NexusManager
from isaaclab_nexus.terrain import NexusTerrain
from agile.rl_env.mdp.terrains import STAND_UP_ROUGH_TERRAIN_G1_CFG as TCFG

REDUCE = bool(int(sys.argv[1])) if len(sys.argv) > 1 else True
MJCF = "/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
NENV, DT = 4, 1 / 200
class _SimCfg: dt = DT; gravity = (0, 0, -9.81); device = "cuda:0"; physics = NexusCfg(contact_reduction=REDUCE)
class _Sim: cfg = _SimCfg(); physics_manager = NexusManager
SimulationContext._instance = _Sim(); NexusManager.initialize(_Sim)

terrain = NexusTerrain(TCFG.replace(num_rows=2, num_cols=3, curriculum=True), NENV, tile=(1, 2), floor_half=0.0)
robot = Articulation(ArticulationCfg(prim_path="/World/envs/env_.*/Robot",
    spawn=NexusMjcfCfg(mjcf_path=MJCF, num_envs=NENV, translation=(0.0, 0.0, terrain.spawn_z(0.10)), auto_floor=False),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=80.0, damping=5.0, effort_limit=200.0)}))
feet = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*ankle_roll_link", history_length=3, track_air_time=True))
robot.set_joint_position_target(torch.zeros(NENV, robot.num_joints, device="cuda"))

torch.cuda.synchronize(); t0 = time.perf_counter()
for i in range(600):
    robot.write_data_to_sim(); NexusManager.step(); robot.update(DT); feet.update(DT)
torch.cuda.synchronize(); el = time.perf_counter() - t0

d = robot.data
foot_z = d.body_link_pos_w.torch[:, feet._ids, 2].min(1).values
gap = foot_z - terrain.heights_at(d.body_link_pos_w.torch[:, feet._ids, :2].mean(1))
fz = feet.data.net_forces_w.torch[..., 2]
mass = sum(robot.data.default_mass.torch[0].tolist()) if hasattr(robot.data, "default_mass") else float("nan")
print(f"contact_reduction={REDUCE} | tris/tile {terrain.num_faces} | {600} steps in {el:.2f}s ({600/el:.0f} steps/s)")
print("  foot - terrain gap (m):", [round(v, 3) for v in gap.tolist()])
print("  foot normal force z (N):", [round(v, 1) for v in fz.sum(1).tolist()])
