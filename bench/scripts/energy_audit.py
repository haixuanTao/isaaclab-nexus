#!/usr/bin/env python3
"""Energy balance audit. Replays a policy, steps the physics one step at a time,
and per physics step accounts for where the robot's mechanical energy comes from:

    dE_mech = W_actuator + W_passive + W_constraint + W_applied   (+ residual)

W_* are MuJoCo's generalized forces dotted with qvel over the step. E_mech is
kinetic + potential energy from the body states. A large positive residual, or
constraint work far above actuator work, is energy the solver injected."""
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--checkpoint", type=str, required=True); parser.add_argument("--envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=400); parser.add_argument("--label", type=str, default="ea")
parser.add_argument("--keep-assist", action="store_true"); parser.add_argument("--assist-scale", type=float, default=None)
AppLauncher.add_app_launcher_args(parser); args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, numpy as np, torch
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_newton.physics import NewtonManager
cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.envs); cfg.seed = 7
# one physics step per env step so every step's forces are observable (policy runs at 200 Hz here)
cfg.decimation = 1; cfg.sim.render_interval = 1
if not args_cli.keep_assist:
    from agile.rl_env.rsl_rl.export_pruning import prepare_training_only_actions_for_evaluation
    print("[assist] removed", prepare_training_only_actions_for_evaluation(cfg)[0], flush=True)
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped
if args_cli.keep_assist and args_cli.assist_scale is not None:
    u.action_manager._terms["lift"].scale_forces(float(args_cli.assist_scale)); print(f"[assist] lift at {args_cli.assist_scale}", flush=True)
from agile.rl_env.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
wenv = RslRlVecEnvWrapper(env); obs, _ = wenv.reset()
policy = torch.jit.load(args_cli.checkpoint, map_location=u.device).eval()
robot = u.scene["robot"]; s = NewtonManager._solver; mjd = s.mjw_data; mjw = s.mjw_model
try:
    import mujoco, warp as wp
    ef = mjw.opt.enableflags
    val = int(ef.numpy()[0]) if hasattr(ef, "numpy") else int(ef)
    val |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    if hasattr(ef, "numpy"): mjw.opt.enableflags = wp.array([val], dtype=ef.dtype, device=ef.device)
    else: mjw.opt.enableflags = val
    HAVE_ENERGY = hasattr(mjd, "energy")
except Exception as exc:
    print("[audit] mujoco energy flag not set:", exc); HAVE_ENERGY = False
print("[audit] mujoco energy available:", HAVE_ENERGY, flush=True)
sensor = u.scene.sensors.get("contact_forces")
def tt(x): return x.torch if hasattr(x, "torch") else x
mass = tt(robot.data.default_mass).to(u.device)                       # (N, B)
inertia = tt(robot.data.default_inertia).to(u.device).reshape(args_cli.envs, -1, 3, 3)
g = 9.81; dt = u.physics_dt if hasattr(u, "physics_dt") else u.sim.get_physics_dt()
def quat_to_mat(q):  # (..., 4) wxyz -> (..., 3, 3)
    w, x, y, z = q.unbind(-1)
    return torch.stack([torch.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], -1),
                        torch.stack([2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], -1),
                        torch.stack([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1)], -2)
def mech_energy():
    v = tt(robot.data.body_com_lin_vel_w) if hasattr(robot.data, "body_com_lin_vel_w") else tt(robot.data.body_lin_vel_w)
    w = tt(robot.data.body_com_ang_vel_w) if hasattr(robot.data, "body_com_ang_vel_w") else tt(robot.data.body_ang_vel_w)
    p = tt(robot.data.body_com_pos_w) if hasattr(robot.data, "body_com_pos_w") else tt(robot.data.body_pos_w)
    R = quat_to_mat(tt(robot.data.body_quat_w)); wb = torch.einsum("nbji,nbj->nbi", R, w)  # world -> body
    ke = 0.5 * (mass * (v * v).sum(-1)).sum(-1) + 0.5 * torch.einsum("nbi,nbij,nbj->nb", wb, inertia, wb).sum(-1)
    pe = (mass * g * p[..., 2]).sum(-1)
    return ke, pe
def gen_power():
    qv = torch.as_tensor(mjd.qvel.numpy(), device=u.device)
    f = {k: torch.as_tensor(getattr(mjd, k).numpy(), device=u.device) for k in ("qfrc_actuator", "qfrc_passive", "qfrc_constraint", "qfrc_applied")}
    return {k: (v * qv).sum(-1) for k, v in f.items()}, qv
W = {k: torch.zeros(args_cli.envs, device=u.device) for k in ("qfrc_actuator", "qfrc_passive", "qfrc_constraint", "qfrc_applied")}
ke0, pe0 = mech_energy(); E0 = ke0 + pe0; Eprev = E0.clone()
worst = []; resid_pos = torch.zeros(args_cli.envs, device=u.device); n_reset = 0; n_acc = 0
acc_max = torch.zeros(args_cli.envs, device=u.device); vprev = None
flight_resid = torch.zeros(args_cli.envs, device=u.device); flight_steps = 0; flight_worst = (0.0, 0, 0)
com_vprev = None; mom_viol = []; mj_ke_err = []
total_mass = mass.sum(-1)
with torch.inference_mode():
    for step in range(args_cli.steps):
        po = obs["policy"] if hasattr(obs, "keys") else obs
        obs, rew, dones, extras = wenv.step(policy(po))
        P, qv = gen_power()
        ke, pe = mech_energy()
        if HAVE_ENERGY:
            e = torch.as_tensor(mjd.energy.numpy(), device=u.device); E = e[:, 0] + e[:, 1]
        else:
            E = ke + pe
        dE = E - Eprev; Eprev = E
        work = {k: v * dt for k, v in P.items()}
        resid = dE - sum(work.values())
        valid = ~dones.bool()                  # a reset rewrites the state: not a physical energy change
        n_reset += int((~valid).sum()); n_acc += int(valid.sum())
        for k in W: W[k] += torch.where(valid, work[k], torch.zeros_like(work[k]))
        E0 = torch.where(valid, E0, E)         # restart the per-env baseline after a reset
        resid_pos += torch.where(valid, resid.clamp(min=0), torch.zeros_like(resid))
        # energy conservation in FLIGHT: no contact on this env this step -> dE must equal actuator+passive work
        Fc_now = tt(sensor.data.net_forces_w).sum(1).norm(dim=-1) if sensor is not None else torch.zeros_like(dE)
        fly = valid & (Fc_now < 1.0)
        flight_steps += int(fly.sum())
        r_f = torch.where(fly, resid, torch.zeros_like(resid)); flight_resid += r_f
        if fly.any():
            jf = int(r_f.abs().argmax())
            if abs(float(r_f[jf])) > abs(flight_worst[0]): flight_worst = (float(r_f[jf]), step, jf)
        # --- cross-checks ---
        if HAVE_ENERGY:
            e = torch.as_tensor(mjd.energy.numpy(), device=u.device)   # (N, 2) potential, kinetic
            mj_ke_err.append(float((e[:, 1] - ke).abs().max()))
        # centre-of-mass momentum: m a_com = F_contact + m g (+ lift). Internal forces cannot move the COM.
        vb = tt(robot.data.body_com_lin_vel_w) if hasattr(robot.data, "body_com_lin_vel_w") else tt(robot.data.body_lin_vel_w)
        com_v = (mass.unsqueeze(-1) * vb).sum(1) / total_mass.unsqueeze(-1)
        if com_vprev is not None:
            a_com = (com_v - com_vprev) / dt
            # exact external force: generalized forces on the root's 3 translational dofs.
            # constraint = all contact forces summed (internal constraints project to zero);
            # actuator/passive on the root must be ~0 for a physical model.
            qc = torch.as_tensor(mjd.qfrc_constraint.numpy(), device=u.device)[:, :3]
            qa = torch.as_tensor(mjd.qfrc_actuator.numpy(), device=u.device)[:, :3]
            qp = torch.as_tensor(mjd.qfrc_passive.numpy(), device=u.device)[:, :3]
            qx = torch.as_tensor(mjd.qfrc_applied.numpy(), device=u.device)[:, :3]
            F_ext = qc + qx + total_mass.unsqueeze(-1) * torch.tensor([0., 0., -g], device=u.device)
            viol = (total_mass.unsqueeze(-1) * a_com - F_ext).norm(dim=-1) / total_mass
            viol = torch.where(valid, viol, torch.zeros_like(viol))
            jj = int(viol.argmax())
            root_act_max = float(qa.norm(dim=-1).max()); root_pas_max = float(qp.norm(dim=-1).max())
            mom_viol.append((float(viol[jj]) / g, float(a_com[jj].norm()) / g, float(qc[jj].norm()), int(mjd.nacon.numpy()[0]), step, jj, root_act_max, root_pas_max))
        com_vprev = com_v.clone()
        # peak body (pelvis) acceleration from consecutive root velocities
        v = tt(robot.data.root_lin_vel_w)
        if vprev is not None:
            a = ((v - vprev).norm(dim=-1) / dt); acc_max = torch.where(valid, torch.maximum(acc_max, a), acc_max)
        vprev = v.clone()
        dEv = torch.where(valid, dE, torch.full_like(dE, -1e9)); j = int(dEv.argmax())
        if valid[j]:
            worst.append((float(dE[j]), float(work["qfrc_actuator"][j]), float(work["qfrc_constraint"][j]), float(work["qfrc_passive"][j]), float(resid[j]), step, j))
E1 = Eprev
print(f"\n[{args_cli.label}] {args_cli.envs} envs x {args_cli.steps} physics steps (dt={dt:.4f})")
tot = lambda k: float(W[k].sum())
dEm = float((E1 - E0).sum())
print(f"[{args_cli.label}] steps accounted={n_acc}  reset steps excluded={n_reset}   peak pelvis acceleration={float(acc_max.max()):.1f} m/s^2 ({float(acc_max.max())/9.81:.1f} g)")
print(f"[{args_cli.label}] TOTAL over all envs:  dE_mech={dEm:10.0f} J | W_actuator={tot('qfrc_actuator'):10.0f} J  W_passive={tot('qfrc_passive'):10.0f} J  W_constraint={tot('qfrc_constraint'):10.0f} J  W_applied={tot('qfrc_applied'):10.0f} J")
print(f"[{args_cli.label}] FLIGHT (no contact) steps={flight_steps}: energy residual sum={float(flight_resid.sum()):.1f} J, worst single step {flight_worst[0]:+.2f} J (step {flight_worst[1]} env {flight_worst[2]})")
print(f"[{args_cli.label}] residual (dE - sum of work) = {dEm - sum(tot(k) for k in W):10.0f} J   positive-residual accumulated = {float(resid_pos.sum()):10.0f} J")
worst.sort(reverse=True)
if mj_ke_err: print(f"[{args_cli.label}] MuJoCo kinetic energy vs my body-based KE: max abs difference {max(mj_ke_err):.2f} J")
mom_viol.sort(reverse=True)
print(f"[{args_cli.label}] COM momentum check -- largest unexplained COM accelerations (g): " + ", ".join(f"{m[0]:.1f}g (a_com {m[1]:.1f}g, root constraint force {m[2]:.0f} N, step {m[4]} env {m[5]})" for m in mom_viol[:4]))
print(f"[{args_cli.label}] steps with unexplained COM acceleration > 2 g: {sum(1 for m in mom_viol if m[0] > 2)} / {len(mom_viol)}   max root-dof actuator force {max(m[6] for m in mom_viol):.1f} N, passive {max(m[7] for m in mom_viol):.1f} N")
print(f"[{args_cli.label}] 8 largest single-step energy gains (one env, one physics step):")
print(f"[{args_cli.label}]   dE(J)   W_act   W_con   W_pas   resid   step env")
for r in worst[:8]: print(f"[{args_cli.label}] {r[0]:7.1f} {r[1]:7.1f} {r[2]:7.1f} {r[3]:7.1f} {r[4]:7.1f}  {r[5]:4d} {r[6]:3d}")
env.close(); simulation_app.close()
