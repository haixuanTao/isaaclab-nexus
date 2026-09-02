"""③ through Isaac Lab's sensor API: terrain + robot + ContactSensor + RayCaster on the Nexus backend."""
import torch
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab_nexus import NexusCfg, NexusMjcfCfg
from isaaclab_nexus.physics.nexus_manager import NexusManager
from isaaclab_nexus.terrain import NexusTerrain
from isaaclab_nexus.sensors.ray_caster import RayCaster
from agile.rl_env.mdp.terrains import STAND_UP_ROUGH_TERRAIN_G1_CFG as TCFG
MJCF = "/workspace/WBC-AGILE/.venv/lib/python3.12/site-packages/newton/examples/assets/nv_humanoid.xml"
NENV, DT = 4, 1 / 200
class _SimCfg: dt = DT; gravity = (0, 0, -9.81); device = "cuda:0"; physics = NexusCfg()
class _Sim: cfg = _SimCfg(); physics_manager = NexusManager
SimulationContext._instance = _Sim(); NexusManager.initialize(_Sim)

terrain = NexusTerrain(TCFG.replace(num_rows=2, num_cols=3, curriculum=True), NENV, tile=(1, 2), floor_half=0.0)
robot = Articulation(ArticulationCfg(prim_path="/World/envs/env_.*/Robot",
    spawn=NexusMjcfCfg(mjcf_path=MJCF, num_envs=NENV, translation=(0.0, 0.0, terrain.spawn_z(0.10)), auto_floor=False),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=80.0, damping=5.0, effort_limit=200.0)}))
feet = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*foot", history_length=3, track_air_time=True))
scan = RayCaster(RayCasterCfg(prim_path="/World/envs/env_.*/Robot/pelvis", pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.0, 0.0)),
                              ray_alignment="yaw", mesh_prim_paths=["/World/ground"], max_distance=5.0))
print(f"factory -> {type(feet).__module__}.ContactSensor | bodies {feet.body_names} | ray caster {type(scan).__module__}")
assert type(feet).__module__.startswith("isaaclab_nexus.")
d = robot.data; robot.set_joint_position_target(torch.zeros(NENV, robot.num_joints, device="cuda"))
air_seen = torch.zeros(NENV, feet.num_sensors, dtype=torch.bool, device="cuda")
for i in range(600):
    robot.write_data_to_sim(); NexusManager.step(); robot.update(DT); feet.update(DT); scan.update(DT)
    if i == 5: air_seen |= feet.data.current_air_time.torch > 0
f = feet.data.net_forces_w.torch
fz = f[..., 2]
print("feet net force z (N) per env:", [[round(v, 1) for v in row] for row in fz.tolist()])
print("current_contact_time (s):", [[round(v, 2) for v in row] for row in feet.data.current_contact_time.torch.tolist()])
print("history shape:", tuple(feet.data.net_forces_w_history.torch.shape), "| air time seen during the fall:", air_seen.any().item())
hit = scan.data.ray_hits_w.torch[:, 0, 2]; pel = d.body_link_pos_w.torch[:, robot.find_bodies("pelvis")[0][0], :2]
print("ray hit z:", [round(v, 3) for v in hit.tolist()], "| grid:", [round(v, 3) for v in terrain.heights_at(pel).tolist()])
foot_z = d.body_link_pos_w.torch[:, feet._ids, 2].min(1).values; gap = foot_z - terrain.heights_at(d.body_link_pos_w.torch[:, feet._ids, :2].mean(1))
print("foot - terrain gap:", [round(v, 3) for v in gap.tolist()])
assert (gap > -0.06).all() and (gap < 0.25).all(), "robot not resting on the terrain"
assert (fz.sum(1) > 50.0).all(), "no contact force on the feet after settling"
assert torch.allclose(hit, terrain.heights_at(pel), atol=1e-4)
print("③ ISAAC SENSORS ON NEXUS OK")
