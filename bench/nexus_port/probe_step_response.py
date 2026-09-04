"""Joint PD step response, Nexus vs PhysX, through AGILE's env (lift off, pushes off, zero actions except one
joint): a 1 rad target step on the left elbow (light link, in the air) and on the left hip pitch; record
q and joint velocity per env step. NEXUS_BACKEND=physx for the stock env."""
import os, torch, numpy as np
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
BACKEND = os.environ.get("NEXUS_BACKEND", "nexus"); TASK = "HeightTracking-G1-v0"
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
env_cfg.scene.num_envs = 4; env_cfg.seed = 3
env_cfg.actions.lift.stiffness_forces = 0.0; env_cfg.actions.lift.damping_forces = 0.0; env_cfg.actions.lift.damping_torques = 0.0
for k in list(vars(env_cfg.events).keys()):
    if k.startswith("push") or k.startswith("apply_external") or "reset_from" in k or "fallen" in k: setattr(env_cfg.events, k, None)
if BACKEND == "nexus":
    from isaaclab_nexus.envs import nexusify; nexusify(env_cfg, "/workspace/bench/nexus_port/g1_29dof_convex64.xml", agent_cfg=agent_cfg)
env = gym.make(TASK, cfg=env_cfg); base = env.unwrapped; robot = base.scene.articulations["robot"]; base.reset()
A = base.action_manager.total_action_dim; jn = robot.joint_names; scale = 0.25
act_term = base.action_manager.get_term("joint_pos") if "joint_pos" in base.action_manager.active_terms else None
zero = torch.zeros(4, A, device=base.device)
def J(name): return jn.index(name)
def T(x): return (x.torch if hasattr(x, "torch") else x)
with torch.no_grad():
    for _ in range(30): base.step(zero)                                   # settle standing
    for jname in ("left_elbow_joint", "left_hip_pitch_joint"):
        j = J(jname); a = zero.clone(); a[:, j] = 1.0 / scale             # +1 rad target step (JointPositionAction scale 0.25)
        q0 = T(robot.data.joint_pos)[0, j].item(); qs, vs, tq = [], [], []
        for i in range(25):
            base.step(a); qs.append(T(robot.data.joint_pos)[0, j].item() - q0); vs.append(T(robot.data.joint_vel)[0, j].item()); tq.append(T(robot.data.applied_torque)[0, j].item())
        for _ in range(30): base.step(zero)
        print(f"[{BACKEND}] {jname}: dq per 20 ms step {np.round(qs[:12], 2).tolist()} | peak |v| {max(abs(v) for v in vs):.1f} rad/s at step {int(np.argmax(np.abs(vs)))} | overshoot {max(qs) - 1.0:+.2f} rad | peak |torque| {max(abs(t) for t in tq):.0f} N.m")
env.close(); app.close()
