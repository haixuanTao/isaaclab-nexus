#!/usr/bin/env python3
"""Does the contact moment have the right sign?

A) Tilt-drop: default pose, pitched by --pitch rad (positive = nose down, feet
   behind the COM), released from --drop m. The ground reaction at the first
   contact (feet/toes) is ahead of or behind the COM; the induced pitch rate must
   match  tau_y = (r_contact - r_com) x F_up.  Reports, at the first contact,
   the horizontal offset of the contact centroid from the COM (x, robot forward)
   and the pitch rate change over the next 10 steps.  Sign rule: contact AHEAD
   of the COM (offset x > 0) -> nose-UP rotation (pitch rate < 0 in the
   convention pitch>0 = nose down).

B) Angular-momentum balance (Newton only): per step, dL/dt about the COM vs
   sum over contacts of (p - com) x F, using the solver's own contact points and
   forces. Prints the correlation and the sign agreement rate."""
import argparse, sys, math
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--label", type=str, default="ms"); parser.add_argument("--envs", type=int, default=8)
parser.add_argument("--drop", type=float, default=0.3); parser.add_argument("--pitch", type=float, default=0.35); parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--rigid", action="store_true", help="lock all joints with very stiff PD so the robot is one rigid body")
AppLauncher.add_app_launcher_args(parser); args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, numpy as np, torch
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from agile.rl_env.rsl_rl.export_pruning import prepare_training_only_actions_for_evaluation
import isaaclab.utils.math as mu
cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.envs); cfg.seed = 3; cfg.decimation = 1; cfg.sim.render_interval = 1
prepare_training_only_actions_for_evaluation(cfg)
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; env.reset()
robot = u.scene["robot"]; sensor = u.scene.sensors.get("contact_forces")
def tt(x): return x.torch if hasattr(x, "torch") else x
mass = tt(robot.data.default_mass).to(u.device); M = mass.sum(-1); dt = u.physics_dt; g = 9.81
N = args_cli.envs; ids = torch.arange(N, device=u.device)
is_newton = False
try:
    from isaaclab_newton.physics import NewtonManager
    is_newton = NewtonManager._solver is not None
except Exception: pass
# --- place: default pose, pitched nose-down about the lateral (y) axis, lifted ---
rp = tt(robot.data.default_root_state)[ids, :7].clone(); rp[:, :3] += u.scene.env_origins[ids]; rp[:, 2] += args_cli.drop
q_pitch = mu.quat_from_euler_xyz(torch.zeros(N, device=u.device), torch.full((N,), args_cli.pitch, device=u.device), torch.zeros(N, device=u.device))
rp[:, 3:7] = mu.quat_mul(q_pitch, rp[:, 3:7])
robot.write_root_pose_to_sim(rp, env_ids=ids); robot.write_root_velocity_to_sim(torch.zeros(N, 6, device=u.device), env_ids=ids)
jp = tt(robot.data.default_joint_pos)[ids].clone(); robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids); u.episode_length_buf[:] = 0
if args_cli.rigid:
    ks = tt(robot.data.joint_stiffness); kd = tt(robot.data.joint_damping)
    robot.write_joint_stiffness_to_sim(torch.full_like(ks, 5000.0)); robot.write_joint_damping_to_sim(torch.full_like(kd, 200.0))
    for a in robot.actuators.values():
        if hasattr(a, "stiffness"): a.stiffness[:] = 5000.0
        if hasattr(a, "damping"): a.damping[:] = 200.0
    print(f"[{args_cli.label}] RIGID: all joints locked with kp=5000, kd=200", flush=True)
body_names = list(robot.body_names)
sensor_names = list(sensor.body_names)
s2r = torch.tensor([body_names.index(n) for n in sensor_names], device=u.device)   # sensor body i -> robot body index
feet_s = [i for i, n in enumerate(sensor_names) if "ankle_roll" in n]
def com():
    p = tt(robot.data.body_com_pos_w) if hasattr(robot.data, "body_com_pos_w") else tt(robot.data.body_pos_w)
    return (mass.unsqueeze(-1) * p).sum(1) / M.unsqueeze(-1), p
def pitch_rate():   # body-frame angular velocity about the lateral axis
    return tt(robot.data.root_ang_vel_b)[:, 1]
first = [None] * N; report = []
L_prev = None; corr_num = 0.0; corr_a = 0.0; corr_b = 0.0; agree = 0; total = 0
with torch.inference_mode():
    for step in range(args_cli.steps):
        env.step(torch.zeros(u.action_space.shape, device=u.device))
        c, pb = com()
        F = tt(sensor.data.net_forces_w)                       # (N, Bs, 3) in SENSOR body order
        Fz = F[..., 2]; touching = Fz.abs() > 5.0
        pbs = pb[:, s2r]                                        # positions in sensor order
        v_ = tt(robot.data.body_com_lin_vel_w) if hasattr(robot.data, "body_com_lin_vel_w") else tt(robot.data.body_lin_vel_w)
        Ltot = (mass.unsqueeze(-1) * torch.cross(pb - c.unsqueeze(1), v_, dim=-1)).sum(1)   # orbital angular momentum about COM
        for e in range(N):
            if first[e] is None and touching[e].any():
                w = Fz[e].abs() * touching[e]; cen = (w.unsqueeze(-1) * pbs[e]).sum(0) / w.sum()
                first_L = float(Ltot[e, 1]); foot_fz = float(Fz[e, feet_s].sum())
                # robot-forward axis from root yaw (x axis of the base rotated by yaw only)
                yaw = mu.euler_xyz_from_quat(tt(robot.data.root_quat_w)[e:e+1])[2]
                fwd = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], -1)[0]
                offset = float(torch.dot(cen - c[e], fwd)); parts = [sensor_names[i] for i in touching[e].nonzero().flatten().tolist()][:4]
                first[e] = (step, offset, float(pitch_rate()[e]), parts, first_L, foot_fz)
                if is_newton and e == 0:
                    mjd_ = NewtonManager._solver.mjw_data; n_ = int(mjd_.nacon.numpy()[0])
                    if n_: print(f"[{args_cli.label}]   (contact solref actually used, first contacts: {mjd_.contact.solref.numpy()[:n_][:2].tolist()})", flush=True)
            elif first[e] is not None and step == first[e][0] + 10:
                report.append((e, first[e][1], float(pitch_rate()[e]) - first[e][2], first[e][3], float(Ltot[e, 1]) - first[e][4], first[e][5]))
        # B) angular momentum balance (Newton)
        if is_newton:
            v = tt(robot.data.body_com_lin_vel_w) if hasattr(robot.data, "body_com_lin_vel_w") else tt(robot.data.body_lin_vel_w)
            r = pb - c.unsqueeze(1)
            L = (mass.unsqueeze(-1) * torch.cross(r, v, dim=-1)).sum(1)           # orbital part about COM (dominant for a falling body)
            cts = NewtonManager._contacts
            if L_prev is not None and cts is not None and cts.force is not None:
                dL = (L - L_prev) / dt
                cnt = int(cts.rigid_contact_count.numpy()[0])
                if cnt:
                    m = NewtonManager.get_model()
                    s0 = cts.rigid_contact_shape0.numpy()[:cnt]; s1 = cts.rigid_contact_shape1.numpy()[:cnt]
                    f = torch.as_tensor(cts.force.numpy()[:cnt][:, :3], device=u.device)
                    p0 = cts.rigid_contact_point0.numpy()[:cnt]; p1 = cts.rigid_contact_point1.numpy()[:cnt]
                    shape_body = m.shape_body.numpy(); body_q = NewtonManager._state_0.body_q.numpy(); body_world = m.body_world.numpy()
                    tau = torch.zeros(N, 3, device=u.device)
                    import warp as wp
                    for k in range(cnt):
                        for shp, pt, sgn in ((int(s0[k]), p0[k], 1.0), (int(s1[k]), p1[k], -1.0)):
                            b = int(shape_body[shp])
                            if b < 0: continue
                            w_ = int(body_world[b])
                            if w_ >= N: continue
                            X = body_q[b]; pw = np.asarray(wp.transform_point(wp.transform(X[:3], X[3:]), wp.vec3(*pt)))
                            pw_t = torch.as_tensor(pw, device=u.device, dtype=torch.float32)
                            tau[w_] += torch.cross(pw_t - c[w_], sgn * f[k], dim=-1)
                    for e in range(N):
                        if tau[e].norm() > 1.0 and dL[e].norm() > 1.0:
                            corr_num += float(torch.dot(tau[e], dL[e])); corr_a += float(tau[e].norm() ** 2); corr_b += float(dL[e].norm() ** 2)
                            agree += int(torch.dot(tau[e], dL[e]) > 0); total += 1
            L_prev = L.clone()
print(f"\n[{args_cli.label}] A) tilt-drop pitch={args_cli.pitch:+.2f} rad (nose down), drop={args_cli.drop} m -- at first contact: contact centroid offset along forward (m), pitch-rate change over next 10 steps (rad/s), bodies")
print(f"[{args_cli.label}]    (sign rule: contact behind COM (offset<0) + upward force -> +tau_y -> total L_y must INCREASE; base pitch rate may differ because the PD-held legs redistribute momentum)")
for e, off, dpr, parts, dL, ffz in report:
    exp_dL = "+" if off < 0 else "-"; got_dL = "+" if dL > 0 else "-"
    print(f"[{args_cli.label}]   env {e}: contact offset={off:+.3f} m, foot Fz at first contact={ffz:+.0f} N -> expected dL_y {exp_dL}; measured dL_y={dL:+.2f} kg m^2/s -> {'OK' if exp_dL == got_dL else 'WRONG SIGN'}   | base d(pitch rate)={dpr:+.2f} rad/s   contacts {parts}")
if is_newton and total:
    cos = corr_num / math.sqrt(corr_a * corr_b)
    print(f"[{args_cli.label}] B) angular-momentum balance about COM (Newton): cosine(tau_contact, dL/dt) = {cos:+.3f}   sign agreement {agree}/{total} steps   |tau|/|dL/dt| ratio = {math.sqrt(corr_a/corr_b):.2f}")
env.close(); simulation_app.close()
