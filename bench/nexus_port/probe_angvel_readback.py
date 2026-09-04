"""Root angular velocity readback vs truth: a free body (G1) in free flight is given a known spin via
write_root_velocity_to_sim; compare data.root_ang_vel_w (and lin vel) with finite differences of the pose."""
import os, numpy as np, torch, nexus3d
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_nexus import NexusCfg, NexusMjcfCfg
from isaaclab_nexus.physics.nexus_manager import NexusManager
import isaaclab.utils.math as mu
G1 = "/workspace/bench/nexus_port/g1_29dof_convex64.xml"; N, DT = 4, 1 / 200
class _SimCfg: dt = DT; gravity = (0, 0, -9.81); device = "cuda:0"; physics = NexusCfg()
class _Sim: cfg = _SimCfg(); physics_manager = NexusManager
SimulationContext._instance = _Sim(); NexusManager.initialize(_Sim); NexusManager.ensure_envs(N)
robot = Articulation(ArticulationCfg(prim_path="/World/envs/env_.*/Robot", init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 20.0)),
    spawn=NexusMjcfCfg(mjcf_path=G1, num_envs=N, auto_floor=False),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=200.0, damping=5.0, effort_limit=100.0)}))
robot.reset(); NexusManager.synchronize()
W = torch.tensor([[0, 0, 10.0], [5.0, 0, 0], [0, 5.0, 0], [3.0, 3.0, 3.0]], device="cuda:0"); V = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]], device="cuda:0")
robot.write_root_velocity_to_sim(torch.cat([V, W], 1)); q0 = robot.data.default_joint_pos.torch.clone()
q_prev = robot.data.root_quat_w.torch.clone(); p_prev = robot.data.root_pos_w.torch.clone()
for i in range(40):
    robot.set_joint_position_target(q0); robot.write_data_to_sim(); NexusManager.step(); robot.update(DT)
    q = robot.data.root_quat_w.torch; p = robot.data.root_pos_w.torch
    dq = mu.quat_mul(q, mu.quat_inv(q_prev)); ang = 2 * torch.atan2(dq[:, :3].norm(dim=-1), dq[:, 3].abs()); w_fd = ang / DT
    axis = dq[:, :3] / (dq[:, :3].norm(dim=-1, keepdim=True) + 1e-9) * torch.sign(dq[:, 3:4]); w_fd_vec = axis * w_fd[:, None]
    v_fd = (p - p_prev) / DT; q_prev = q.clone(); p_prev = p.clone()
    if i in (0, 1, 5, 20, 39):
        rep_w = robot.data.root_ang_vel_w.torch; rep_v = robot.data.root_lin_vel_w.torch
        for e in range(N):
            print(f"step {i:2d} env {e}: set w {W[e].tolist()} | reported w {rep_w[e].cpu().numpy().round(2)} |w| {rep_w[e].norm():.2f} | FD w {w_fd_vec[e].cpu().numpy().round(2)} |w| {w_fd[e]:.2f} | reported v {rep_v[e].cpu().numpy().round(2)} FD v {v_fd[e].cpu().numpy().round(2)}")
