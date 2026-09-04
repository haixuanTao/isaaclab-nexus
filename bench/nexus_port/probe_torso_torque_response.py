"""Whole-body response to an off-root wrench: standing G1 (AGILE env, lift disabled, zero actions), apply a
100 N.m torque about x (and separately a 300 N upward force at +0.5 m) at torso_link for ONE env step; report
the root's angular/linear velocity change. NEXUS_BACKEND=physx for the stock env."""
import os, torch
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
BACKEND = os.environ.get("NEXUS_BACKEND", "nexus"); TASK = "HeightTracking-G1-v0"
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
env_cfg.scene.num_envs = 8; env_cfg.seed = 3
env_cfg.actions.lift.stiffness_forces = 0.0; env_cfg.actions.lift.damping_forces = 0.0; env_cfg.actions.lift.damping_torques = 0.0
for k in list(vars(env_cfg.events).keys()):                    # no pushes / random forces
    if k.startswith("push") or k.startswith("apply_external"): setattr(env_cfg.events, k, None)
if BACKEND == "nexus":
    from isaaclab_nexus.envs import nexusify; nexusify(env_cfg, "/workspace/bench/nexus_port/g1_29dof_convex64.xml", agent_cfg=agent_cfg)
env = gym.make(TASK, cfg=env_cfg); base = env.unwrapped; robot = base.scene.articulations["robot"]
base.reset(); tid = robot.find_bodies("torso_link")[0]
zero = torch.zeros(8, base.action_manager.total_action_dim, device=base.device)
def W(): d = robot.data; w = d.root_ang_vel_w; v = d.root_lin_vel_w; return (w.torch if hasattr(w, "torch") else w).clone(), (v.torch if hasattr(v, "torch") else v).clone()
with torch.no_grad():
    for _ in range(25): base.step(zero)                         # settle
    w0, v0 = W(); N = 8
    F = torch.zeros(N, 1, 3, device=base.device); T = torch.zeros(N, 1, 3, device=base.device); T[:4, 0, 0] = 100.0; F[4:, 0, 2] = 300.0
    robot.set_external_force_and_torque(F, T, body_ids=tid); base.step(zero); w1, v1 = W()
    robot.set_external_force_and_torque(torch.zeros_like(F), torch.zeros_like(T), body_ids=tid)
    dw = (w1 - w0); dv = (v1 - v0)
    print(f"[{BACKEND}] 100 N.m about x at torso for one env step (0.02 s): root d(omega) mean {dw[:4].mean(0).cpu().numpy().round(2)} rad/s | root d(v) {dv[:4].mean(0).cpu().numpy().round(3)} m/s")
    print(f"[{BACKEND}] 300 N up at torso (offset none) for one env step: root d(v) mean {dv[4:].mean(0).cpu().numpy().round(3)} m/s | d(omega) {dw[4:].mean(0).cpu().numpy().round(2)} rad/s | expected dv_z ~ 300/35kg*0.02 = 0.17 minus gravity effects")
env.close(); app.close()
