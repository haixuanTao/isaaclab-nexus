"""③ terrain v2: spawn ABOVE the tile via insert translation, loader floor off, then let the robot settle.
Feet must rest on the terrain surface (height grid), and the RayCaster must report that height."""
import torch
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab_nexus import NexusCfg, NexusMjcfCfg
from isaaclab_nexus.physics.nexus_manager import NexusManager
from isaaclab_nexus.terrain import NexusTerrain
from isaaclab_nexus.sensors.ray_caster import RayCaster
from agile.rl_env.mdp.terrains import STAND_UP_ROUGH_TERRAIN_G1_CFG as TCFG
MJCF="/workspace/WBC-AGILE/.venv/lib/python3.12/site-packages/newton/examples/assets/nv_humanoid.xml"
NENV, DT = 4, 1/200
class _SimCfg: dt=DT; gravity=(0,0,-9.81); device="cuda:0"; physics=NexusCfg()
class _Sim: cfg=_SimCfg(); physics_manager=NexusManager
SimulationContext._instance=_Sim(); NexusManager.initialize(_Sim)
terrain=NexusTerrain(TCFG.replace(num_rows=2,num_cols=3,curriculum=True), NENV, tile=(1,2), floor_half=0.0)
zs=terrain.spawn_z(0.10)
print(f"tile faces {terrain.num_faces}, relief {terrain.height.min():.3f}..{terrain.height.max():.3f}, spawn clearance {zs:.3f}")
robot=Articulation(ArticulationCfg(prim_path="/World/envs/env_.*/Robot",
    spawn=NexusMjcfCfg(mjcf_path=MJCF, num_envs=NENV, translation=(0.0,0.0,zs), auto_floor=False),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=80.0, damping=5.0, effort_limit=200.0)}))
scanner=RayCaster(RayCasterCfg(prim_path="/World/envs/env_.*/Robot/pelvis", pattern_cfg=patterns.GridPatternCfg(resolution=0.05,size=(0.0,0.0)), ray_alignment="yaw", mesh_prim_paths=["/World/ground"], max_distance=5.0))
d=robot.data; robot.set_joint_position_target(torch.zeros(NENV, robot.num_joints, device="cuda"))
for _ in range(600): robot.write_data_to_sim(); NexusManager.step()
robot.update(DT); scanner.update(DT)
feet,_=robot.find_bodies(".*foot")
foot_z=d.body_link_pos_w.torch[:,feet,2].min(1).values; foot_xy=d.body_link_pos_w.torch[:,feet,:2].mean(1)
h=terrain.heights_at(foot_xy); gap=foot_z-h
print("terrain h under feet:", [round(v,3) for v in h.tolist()], "| lowest foot z:", [round(v,3) for v in foot_z.tolist()])
print("foot - terrain gap  :", [round(v,3) for v in gap.tolist()])
hit=scanner.data.ray_hits_w.torch[:,0,2]; pel=d.body_link_pos_w.torch[:, robot.find_bodies("pelvis")[0][0], :2]
print("ray hit z under pelvis:", [round(v,3) for v in hit.tolist()], "| grid:", [round(v,3) for v in terrain.heights_at(pel).tolist()])
assert (gap > -0.06).all() and (gap < 0.20).all(), "feet not resting on the terrain"
assert torch.allclose(hit, terrain.heights_at(pel), atol=1e-4), "ray caster disagrees with height grid"
print("③ terrain + ray caster OK")
