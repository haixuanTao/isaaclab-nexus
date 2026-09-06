"""Per-joint step response, engine PD vs Python PD: bare G1 in free flight (no gravity), env j steps joint j's target by
+0.3 rad; record every joint's displacement after 1 and 4 control steps. NEXUS_ENGINE_PD=1|0 -> npz for comparison."""
import os, sys, numpy as np, torch, nexus3d
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_nexus import NexusCfg, NexusMjcfCfg
from isaaclab_nexus.physics.nexus_manager import NexusManager
G1 = "/workspace/bench/nexus_port/g1_29dof_convex64.xml"; N, DT = int(os.environ.get("N", 29)), 1 / 200
GRAV, FLOOR, Z = float(os.environ.get("GRAV", 0)), os.environ.get("FLOOR", "0") == "1", float(os.environ.get("Z", 5.0))
KP, KD, EFF, VEL = float(os.environ.get("KP", 50)), float(os.environ.get("KD", 1)), float(os.environ.get("EFF", 1000)), float(os.environ.get("VEL", 1000))
class _SimCfg: dt = DT; gravity = (0, 0, GRAV); device = "cuda:0"; physics = NexusCfg()
class _Sim: cfg = _SimCfg(); physics_manager = NexusManager
SimulationContext._instance = _Sim(); NexusManager.initialize(_Sim); NexusManager.ensure_envs(N)
actuators = {"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=KP, damping=KD, effort_limit_sim=EFF, velocity_limit_sim=VEL)}
if os.environ.get("ACT") == "agile":                      # AGILE's real G1 actuator models (DelayedDCMotor, per-joint gains/limits)
    import importlib, copy; U = importlib.import_module(os.environ.get("AGILE_G1_MODULE", "agile.assets.unitree_g1"))
    cfgs = [v for v in vars(U).values() if isinstance(v, ArticulationCfg)]
    src = [c for c in cfgs if any("waist" in k for k in c.actuators)] or cfgs
    actuators = copy.deepcopy(src[0].actuators); print("AGILE actuators:", {k: type(v).__name__ for k, v in actuators.items()})
rot = (0.7071, 0.0, 0.0, 0.7071) if os.environ.get("LIE", "0") == "1" else (0.0, 0.0, 0.0, 1.0)   # xyzw: lying on the side
robot = Articulation(ArticulationCfg(prim_path="/World/envs/env_.*/Robot", init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, Z), rot=rot),
    spawn=NexusMjcfCfg(mjcf_path=G1, num_envs=N, auto_floor=FLOOR), actuators=actuators))
robot.reset(); NexusManager.synchronize(); print("root pose:", robot.data.root_pos_w.torch[0].tolist(), robot.data.root_quat_w.torch[0].tolist())
q0 = robot.data.default_joint_pos.torch.clone(); tgt = q0.clone(); tgt[torch.arange(29), torch.arange(29)] += 0.3
out = {}
for i in range(4):
    robot.set_joint_position_target(tgt); robot.write_data_to_sim()
    for _ in range(4): NexusManager.step()
    NexusManager.synchronize(); robot.update(DT)
    out[f"dq{i+1}"] = (robot.data.joint_pos.torch - q0).cpu().numpy(); out[f"rw{i+1}"] = robot.data.root_ang_vel_w.torch.cpu().numpy()
np.savez(sys.argv[1], names=np.array(robot.joint_names), **out); print("saved", sys.argv[1])
