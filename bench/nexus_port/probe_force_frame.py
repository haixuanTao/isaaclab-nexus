"""Frame of the root free-joint generalized force. Zero-g, robot floating: +X force must give +vx in
world if the force slots are world-frame; also print the root link quat and the root JOINT_ROT quad."""
import os, torch, nexus3d
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_nexus import NexusCfg, NexusMjcfCfg
from isaaclab_nexus.physics.nexus_manager import NexusManager
G1 = "/workspace/bench/nexus_port/g1_29dof_convex64.xml"; N, DT = 4, 1 / 200
class _SimCfg: dt = DT; gravity = (0, 0, 0); device = "cuda:0"; physics = NexusCfg()
class _Sim: cfg = _SimCfg(); physics_manager = NexusManager
SimulationContext._instance = _Sim(); NexusManager.initialize(_Sim)
robot = Articulation(ArticulationCfg(prim_path="/World/envs/env_.*/Robot", init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 3.0)),
    spawn=NexusMjcfCfg(mjcf_path=G1, num_envs=N, auto_floor=False), actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=150.0, damping=5.0, effort_limit=139.0)}))
robot.reset(); q0 = robot.data.default_joint_pos.torch.clone(); torso = robot.find_bodies("torso_link")[0]; lay = robot._lay
NexusManager.synchronize(); print("root_link_quat_w (wxyz) at rest:", [round(x, 3) for x in robot.data.root_link_quat_w.torch[0].tolist()])
for k in sorted(lay): 
    if "ROT" in k or "COORD" in k: print(f"  layout {k} = {lay[k]}")
def run(label, F, T, steps=10):
    robot.reset(); pose = robot.data.default_root_pose.torch.clone(); robot.write_root_pose_to_sim(pose); robot.write_joint_state_to_sim(q0, torch.zeros_like(q0)); robot.write_root_velocity_to_sim(torch.zeros(N, 6, device="cuda:0")); NexusManager.synchronize()
    f = torch.zeros(N, 1, 3, device="cuda:0"); t = torch.zeros(N, 1, 3, device="cuda:0"); f[:, 0] = torch.tensor(F, device="cuda:0"); t[:, 0] = torch.tensor(T, device="cuda:0")
    for i in range(steps):
        robot.set_joint_position_target(q0); robot.set_external_force_and_torque(f, t, body_ids=torso); robot.write_data_to_sim(); NexusManager.step(); robot.update(DT)
    v = robot.data.root_lin_vel_w.torch.mean(0); w = robot.data.root_ang_vel_w.torch.mean(0)
    print(f"[{label}] after {steps} steps: v=({float(v[0]):+.2f},{float(v[1]):+.2f},{float(v[2]):+.2f})  w=({float(w[0]):+.2f},{float(w[1]):+.2f},{float(w[2]):+.2f})")
run("+200 N x-force ", (200.0, 0, 0), (0, 0, 0)); run("+200 N y-force ", (0, 200.0, 0), (0, 0, 0)); run("+200 N z-force ", (0, 0, 200.0), (0, 0, 0))
run("+50 Nm x-torque", (0, 0, 0), (50.0, 0, 0)); run("+50 Nm y-torque", (0, 0, 0), (0, 50.0, 0)); run("+50 Nm z-torque", (0, 0, 0), (0, 0, 50.0))
