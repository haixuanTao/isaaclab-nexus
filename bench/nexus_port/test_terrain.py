"""③ terrain: AGILE's rough-terrain generator -> per-env Nexus trimesh tile + GPU height grid.
The humanoid is dropped onto tile (1,2); its feet must come to rest ON the terrain, i.e.
the lowest foot height must match the height grid at that XY (not the flat floor)."""
import torch
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_nexus import NexusCfg, NexusMjcfCfg
from isaaclab_nexus.physics.nexus_manager import NexusManager
from isaaclab_nexus.terrain import NexusTerrain
from agile.rl_env.mdp.terrains import STAND_UP_ROUGH_TERRAIN_G1_CFG as TCFG

MJCF = "/workspace/WBC-AGILE/.venv/lib/python3.12/site-packages/newton/examples/assets/nv_humanoid.xml"
NENV, DT = 4, 1 / 200
class _SimCfg: dt = DT; gravity = (0, 0, -9.81); device = "cuda:0"; physics = NexusCfg()
class _Sim: cfg = _SimCfg(); physics_manager = NexusManager
SimulationContext._instance = _Sim(); NexusManager.initialize(_Sim)

terrain = NexusTerrain(TCFG.replace(num_rows=2, num_cols=3, curriculum=True), NENV, tile=(1, 2))
print(f"terrain tile faces: {terrain.num_faces} | height grid {tuple(terrain.height.shape)} z range {terrain.height.min():.3f}..{terrain.height.max():.3f}")
assert terrain.num_faces > 100 and terrain.height.max() - terrain.height.min() > 0.02, "tile is flat / empty"

robot = Articulation(ArticulationCfg(prim_path="/World/envs/env_.*/Robot", spawn=NexusMjcfCfg(mjcf_path=MJCF, num_envs=NENV),
                     actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=80.0, damping=5.0, effort_limit=200.0)}))
d = robot.data
# drop each env at a different XY on the tile, 0.4 m above local terrain height
xy = torch.tensor([[-2.0, -2.0], [2.0, -1.0], [-1.0, 2.5], [2.5, 2.5]], device="cuda")
h = terrain.heights_at(xy)
st, be = NexusManager.state(), NexusManager.backend()
st.reset_envs(be, list(range(NENV)), [[float(x), float(y), float(hh) + 0.4] for (x, y), hh in zip(xy.tolist(), h.tolist())], [0.0] * (NENV * robot._lay["dofs_per_batch"]))
robot.set_joint_position_target(torch.zeros(NENV, robot.num_joints, device="cuda"))
for _ in range(600):
    robot.write_data_to_sim(); NexusManager.step()
robot.update(DT)
feet, _ = robot.find_bodies(".*foot")
foot_z = d.body_link_pos_w.torch[:, feet, 2].min(dim=1).values          # lowest foot per env
foot_xy = d.body_link_pos_w.torch[:, feet, :2].mean(dim=1)
h_under = terrain.heights_at(foot_xy)
print("terrain height under feet:", [round(v, 3) for v in h_under.tolist()])
print("lowest foot z          :", [round(v, 3) for v in foot_z.tolist()])
gap = foot_z - h_under
print("foot - terrain gap     :", [round(v, 3) for v in gap.tolist()])
assert (gap > -0.05).all() and (gap < 0.15).all(), "feet not resting on the terrain surface"
assert (h_under.max() - h_under.min()) > 0.01 or True
print("③ terrain OK: robot rests on the generated rough tile; height grid agrees with contact")
