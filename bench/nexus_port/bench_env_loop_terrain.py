"""④ throughput with terrain + Isaac sensors at scale: NexusTerrain tile per env (0.25 m collider), humanoid,
ContactSensor on the feet, RayCaster under the pelvis; write targets -> step -> update -> read obs each step."""
import sys, time, torch
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
NENV = int(sys.argv[1]) if len(sys.argv) > 1 else 4096; DT = 1 / 200; STEPS = 200
class _SimCfg: dt = DT; gravity = (0, 0, -9.81); device = "cuda:0"; physics = NexusCfg()
class _Sim: cfg = _SimCfg(); physics_manager = NexusManager
SimulationContext._instance = _Sim(); NexusManager.initialize(_Sim)
t0 = time.perf_counter()
terrain = NexusTerrain(TCFG.replace(num_rows=2, num_cols=3, curriculum=True), NENV, tile=(1, 2), floor_half=0.0)
robot = Articulation(ArticulationCfg(prim_path="/World/envs/env_.*/Robot",
    spawn=NexusMjcfCfg(mjcf_path=MJCF, num_envs=NENV, translation=(0.0, 0.0, terrain.spawn_z(0.10)), auto_floor=False),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=80.0, damping=5.0, effort_limit=200.0)}))
feet = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*foot", history_length=3, track_air_time=True))
scan = RayCaster(RayCasterCfg(prim_path="/World/envs/env_.*/Robot/pelvis", pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.0, 1.0)),
                              ray_alignment="yaw", mesh_prim_paths=["/World/ground"], max_distance=5.0))
torch.cuda.synchronize(); print(f"setup: {NENV} envs, {terrain.num_faces} collider tris/env, {scan.num_rays} rays/env in {time.perf_counter()-t0:.1f}s | mem {torch.cuda.memory_allocated()/2**30:.2f} GiB alloc, {torch.cuda.mem_get_info()[1]/2**30-torch.cuda.mem_get_info()[0]/2**30:.2f} GiB used")
d = robot.data; tgt = torch.zeros(NENV, robot.num_joints, device="cuda")
def loop(n):
    for i in range(n):
        robot.set_joint_position_target(tgt + 0.1 * torch.sin(torch.tensor(i * 0.05)))
        robot.write_data_to_sim(); NexusManager.step(); robot.update(DT); feet.update(DT); scan.update(DT)
        obs = torch.cat([d.joint_pos.torch, d.joint_vel.torch, d.root_lin_vel_w.torch, feet.data.net_forces_w.torch.flatten(1), scan.data.ray_hits_w.torch[..., 2]], 1)
    return obs
loop(20); torch.cuda.synchronize(); t = time.perf_counter(); obs = loop(STEPS); torch.cuda.synchronize(); el = time.perf_counter() - t
print(f"{NENV} envs x {STEPS} steps: {el:.2f}s -> {STEPS/el:.1f} steps/s, {NENV*STEPS/el/1e6:.3f} M env-steps/s | obs {tuple(obs.shape)} finite={bool(torch.isfinite(obs).all())}")
gap = d.body_link_pos_w.torch[:, feet._ids, 2].min(1).values - terrain.heights_at(d.body_link_pos_w.torch[:, feet._ids, :2].mean(1))
print(f"foot-terrain gap after {STEPS+20} steps: median {gap.median().item():+.3f} m, min {gap.min().item():+.3f}, max {gap.max().item():+.3f} | feet in contact: {(feet.data.net_forces_w.torch[...,2] > 1).float().mean().item()*100:.0f}%")
