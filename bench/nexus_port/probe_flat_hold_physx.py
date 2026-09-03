"""Same flat-floor PD hold on PhysX: AGILE's G1 USD, ground plane, ImplicitActuator PD toward the
default pose at the same gains as probe_flat_hold_direct.py. The engine-vs-pose control."""
import torch
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import Articulation
from isaaclab.actuators import ImplicitActuatorCfg
from agile.rl_env.assets.robots import unitree_g1 as g1mod
base_cfg = next(v for k, v in vars(g1mod).items() if k.startswith("G1_29DOF") and hasattr(v, "spawn"))
N, DT = 64, 1 / 200
sim = SimulationContext(SimulationCfg(dt=DT, device="cuda:0"))
sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)).func("/World/ground", sim_utils.GroundPlaneCfg())
import omni.usd
from pxr import UsdGeom, Gf
stage = omni.usd.get_context().get_stage(); UsdGeom.Xform.Define(stage, "/World/envs")
for e in range(N):
    x = UsdGeom.Xform.Define(stage, f"/World/envs/env_{e}"); UsdGeom.XformCommonAPI(x).SetTranslate(Gf.Vec3d(3.0 * (e % 8), 3.0 * (e // 8), 0.0))
cfg = base_cfg.replace(prim_path="/World/envs/env_.*/Robot", actuators={
    "legs": ImplicitActuatorCfg(joint_names_expr=[".*_hip_.*", ".*_knee_joint"], stiffness=150.0, damping=5.0, effort_limit=139.0),
    "feet": ImplicitActuatorCfg(joint_names_expr=[".*_ankle_.*"], stiffness=40.0, damping=2.0, effort_limit=50.0),
    "rest": ImplicitActuatorCfg(joint_names_expr=["waist_.*", ".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*"], stiffness=40.0, damping=2.0, effort_limit=25.0)})
robot = Articulation(cfg); sim.reset(); robot.reset(); q0 = robot.data.default_joint_pos.torch.clone(); out = []
for i in range(600):
    robot.set_joint_position_target(q0); robot.write_data_to_sim(); sim.step(render=False); robot.update(DT)
    if i in (24, 99, 199, 399, 599):
        z = robot.data.root_link_pos_w.torch[:, 2]; out.append(f"t={(i+1)*DT:.2f}s z {float(z.mean()):.2f} up {float((z>0.6).float().mean())*100:.0f}%")
tq = robot.data.applied_torque.torch.abs()
print(f"[physx FLAT ground plane, friction 1.0] " + " | ".join(out) + f" | |torque| median {float(tq.median()):.1f} max {float(tq.max()):.1f}")
app.close()
