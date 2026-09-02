"""Factory-level test of the Nexus backend's read AND write paths through Isaac Lab's
own `Articulation` API (no Kit, SimulationContext shimmed as before).

Covers: names + regex lookup, ProxyArray data (`.torch`), flat joint_pos/vel with the
floating base excluded, position targets via actuator gains from cfg, effort targets,
batched reset, root-pose write, and loud failure on the unimplemented surface.
"""
import torch
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_nexus import NexusCfg, NexusMjcfCfg
from isaaclab_nexus.physics.nexus_manager import NexusManager

MJCF = "/workspace/WBC-AGILE/.venv/lib/python3.12/site-packages/newton/examples/assets/nv_humanoid.xml"
NENV, DT = 8, 1.0 / 200.0

class _SimCfg: dt = DT; gravity = (0.0, 0.0, -9.81); device = "cuda:0"; physics = NexusCfg()
class _Sim:    cfg = _SimCfg(); physics_manager = NexusManager
SimulationContext._instance = _Sim(); NexusManager.initialize(_Sim)

cfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=NexusMjcfCfg(mjcf_path=MJCF, num_envs=NENV),
    actuators={
        "legs": ImplicitActuatorCfg(joint_names_expr=[".*hip.*", ".*knee.*", ".*ankle.*"], stiffness=80.0, damping=5.0, effort_limit=200.0),
        # arms under pure effort control (Isaac semantic: stiffness/damping 0) so the effort path is observable
        "arms": ImplicitActuatorCfg(joint_names_expr=[".*shoulder.*", ".*elbow.*"], stiffness=0.0, damping=0.0, effort_limit=80.0),
        "waist": ImplicitActuatorCfg(joint_names_expr=["abdomen.*"], stiffness=60.0, damping=4.0, effort_limit=150.0),
    },
)
robot = Articulation(cfg)
assert type(robot).__module__.startswith("isaaclab_nexus.")
print(f"factory -> {type(robot).__module__}.{type(robot).__name__} | envs {robot.num_instances} bodies {robot.num_bodies} joints {robot.num_joints}")
print("joint names:", robot.joint_names)
assert robot.num_joints == 21 and robot.num_bodies == 22

# ---- names + regex ----
ids, names = robot.find_joints(["left_.*"])
print(f"find_joints('left_.*') -> {names}")
assert set(names) == {n for n in robot.joint_names if n.startswith("left_")}
bids, bnames = robot.find_bodies(".*foot|.*shin")
print(f"find_bodies('.*foot|.*shin') -> {bnames}")

# ---- data via ProxyArray (AGILE reads data.joint_pos.torch[...]) ----
d = robot.data
q = d.joint_pos.torch; v = d.joint_vel.torch
print("joint_pos.torch", tuple(q.shape), q.device, "| joint_vel", tuple(v.shape), "| root_pos_w", tuple(d.root_link_pos_w.torch.shape))
assert q.shape == (NENV, 21) and v.shape == (NENV, 21)
assert d.root_link_quat_w.torch.norm(dim=-1).sub(1).abs().max() < 1e-4

def step(n):
    for _ in range(n):
        robot.write_data_to_sim(); NexusManager.step()
    robot.update(DT)

# ---- position targets (PD gains from cfg.actuators) ----
target = torch.full((NENV, 21), 0.25, device="cuda")
robot.set_joint_position_target(target)
q0 = d.joint_pos.torch.clone(); step(200); q1 = d.joint_pos.torch
e0, e1 = (q0 - target).abs().mean().item(), (q1 - target).abs().mean().item()
print(f"position targets: mean|q-q*| {e0:.3f} -> {e1:.3f} rad after 1.0 s")
assert e1 < 0.5 * e0

# ---- effort targets on the left elbow only (arms have zero PD gains) ----
robot.set_joint_position_target(torch.zeros(NENV, 21, device="cuda")); step(50)
robot.reset(); NexusManager.synchronize()
kid, _ = robot.find_joints("left_elbow")
tau = torch.zeros(NENV, 21, device="cuda"); tau[:, kid] = 10.0
robot.set_joint_effort_target(tau)
vb = d.joint_vel.torch[:, kid].mean().item(); step(5); va = d.joint_vel.torch[:, kid].mean().item()
print(f"effort on left_elbow (no PD): v {vb:+.3f} -> {va:+.3f} rad/s")
assert va - vb > 0.05
robot.set_joint_effort_target(torch.zeros(NENV, 21, device="cuda")); step(1)

# ---- reset a subset + root pose write ----
step(150)
z_fallen = d.root_link_pos_w.torch[:, 2].clone()
robot.reset(env_ids=[0, 3]); z = d.root_link_pos_w.torch[:, 2]
z0 = d.default_root_pose.torch[:, 2]
print(f"reset envs [0,3]: z env0 {z_fallen[0]:.3f}->{z[0]:.3f} (template {z0[0]:.3f}); env1 untouched {z_fallen[1]:.3f}->{z[1]:.3f}")
assert abs(z[0] - z0[0]) < 0.02 and abs(z[3] - z0[3]) < 0.02 and torch.isclose(z[1], z_fallen[1])
pose = d.default_root_pose.torch[[2]].clone(); pose[0, 0] += 3.0
robot.write_root_pose_to_sim(pose, env_ids=[2])
print(f"write_root_pose_to_sim(env2, x+3): root x = {d.root_link_pos_w.torch[2,0]:.3f}")
assert abs(d.root_link_pos_w.torch[2, 0] - pose[0, 0]) < 0.02

# ---- unimplemented surface fails loudly ----
try:
    robot.write_root_velocity_to_sim(torch.zeros(NENV, 6, device="cuda")); raise SystemExit("expected NotImplementedError")
except NotImplementedError as e:
    print("loud failure:", str(e)[:64])
print("ISAAC LAB API -> NEXUS: READ + WRITE PATH OK")
