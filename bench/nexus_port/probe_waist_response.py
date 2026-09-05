"""Does an external force on torso_link act through the waist joints? Standing G1 (AGILE env, zero gains on the
waist would be ideal but keep AGILE's), push 150 N forward (+x) at torso_link for 0.3 s via the articulation API
directly (no env.step, so the action term cannot overwrite the composer); report the change of the waist pitch
joint and of the torso tilt, on NEXUS_BACKEND=nexus|physx."""
import os, torch
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
BACKEND = os.environ.get("NEXUS_BACKEND", "nexus"); TASK = "HeightTracking-G1-v0"
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
env_cfg.scene.num_envs = 4; env_cfg.seed = 3; env_cfg.actions.lift.stiffness_forces = 0.0; env_cfg.actions.lift.damping_forces = 0.0; env_cfg.actions.lift.damping_torques = 0.0
for k in list(vars(env_cfg.events).keys()):
    if k.startswith("push") or k.startswith("apply_external") or "reset_from" in k or "fallen" in k: setattr(env_cfg.events, k, None)
if BACKEND == "nexus":
    from isaaclab_nexus.envs import nexusify; nexusify(env_cfg, "/workspace/bench/nexus_port/g1_29dof_convex64.xml", agent_cfg=agent_cfg)
env = gym.make(TASK, cfg=env_cfg); base = env.unwrapped; robot = base.scene.articulations["robot"]; base.reset()
def T(x): return x.torch if hasattr(x, "torch") else x
zero = torch.zeros(4, base.action_manager.total_action_dim, device=base.device)
with torch.no_grad():
    for _ in range(30): base.step(zero)                                   # settle standing under AGILE's PD
    tid = robot.find_bodies("torso_link")[0]; wj = robot.joint_names.index("waist_pitch_joint"); hj = robot.joint_names.index("left_hip_pitch_joint")
    q0 = T(robot.data.joint_pos).clone(); tq0 = T(robot.data.body_link_quat_w)[:, tid[0]].clone(); rz0 = T(robot.data.root_link_pos_w)[:, 2].clone()
    F = torch.zeros(4, 1, 3, device=base.device); F[:, 0, 0] = 150.0; Tq = torch.zeros_like(F)
    dt = base.physics_dt; n = int(0.3 / dt)
    for i in range(n):
        robot.set_joint_position_target(robot.data.default_joint_pos.torch if hasattr(robot.data.default_joint_pos, "torch") else robot.data.default_joint_pos)
        robot.set_external_force_and_torque(F, Tq, body_ids=tid); robot.write_data_to_sim(); base.sim.step(); robot.update(dt)
    q1 = T(robot.data.joint_pos); tq1 = T(robot.data.body_link_quat_w)[:, tid[0]]; rz1 = T(robot.data.root_link_pos_w)[:, 2]
    if hasattr(robot, "_wrench_tau"):
        jn = robot.joint_names; wt = robot._wrench_tau[0]
        print("   [nexus] joints loaded by the torso force (env 0):", {jn[i]: round(float(wt[i]), 1) for i in range(len(jn)) if abs(float(wt[i])) > 0.5})
        wi = [jn.index(n) for n in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")]
        print("   [nexus] waist axes (local):", robot._jax_local[wi].cpu().numpy().round(2).tolist(), "| anchors:", robot._janchor_local[wi].cpu().numpy().round(3).tolist(), "| jlink:", robot._jlink[wi].tolist(), "torso body idx:", tid[0])
        import warp as wp; comp = robot.permanent_wrench_composer
        print("   [nexus] composer active:", comp.active, "| local force torso:", wp.to_torch(comp.local_force_b)[0, tid[0]].cpu().numpy().round(1), "| global:", wp.to_torch(comp.global_force_w)[0, tid[0]].cpu().numpy().round(1), "| com torso:", T(robot.data.body_com_pos_w)[0, tid[0]].cpu().numpy().round(3), "| torso link pos:", T(robot.data.body_link_pos_w)[0, tid[0]].cpu().numpy().round(3))
        print("   [nexus] ancestor joints of torso_link:", [jn[i] for i in range(len(jn)) if robot._anc[i, tid[0]] > 0], "| root effort rows:", robot._effort[robot._root_cols, 0].cpu().numpy().round(1))
    up0 = 1 - 2 * (tq0[:, 0] ** 2 + tq0[:, 1] ** 2); up1 = 1 - 2 * (tq1[:, 0] ** 2 + tq1[:, 1] ** 2)
    print(f"[{BACKEND}] 150 N +x at torso_link for 0.3 s: d(waist_pitch) {float((q1 - q0)[:, wj].mean()):+.3f} rad | d(left_hip_pitch) {float((q1 - q0)[:, hj].mean()):+.3f} rad | torso tilt cos {float(up0.mean()):.3f} -> {float(up1.mean()):.3f} | root dz {float((rz1 - rz0).mean()):+.3f} m")
env.close(); app.close()
