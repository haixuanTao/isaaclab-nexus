"""Per-joint torque-direction test through AGILE's real actuator path on the Nexus backend.
Gravity off, robot in the air. Env e perturbs ONLY joint (e mod J)'s position target by +delta;
after K physics steps that joint must have moved toward the target and every other joint must
have moved far less. Any joint moving the wrong way = sign/axis error; motion appearing on a
different joint = DOF-map error."""
import os, sys, torch
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_nexus.envs import nexusify
from isaaclab_nexus.physics.nexus_manager import NexusManager
TASK = "HeightTracking-G1-v0"; G1 = os.environ.get("NEXUS_G1_MJCF", "/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml")
DELTA, K = 0.3, 40
cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); cfg.scene.num_envs = 64; cfg.seed = 42
cfg.sim.gravity = (0.0, 0.0, 0.0)
for name in ("push_robot", "randomize_physics_material", "randomize_base_com"):
    if getattr(cfg.events, name, None) is not None: setattr(cfg.events, name, None)
nexusify(cfg, G1); env = gym.make(TASK, cfg=cfg).unwrapped; env.reset()
robot = env.scene.articulations["robot"]; J = robot.num_joints; N = 64
# lift the robots so nothing touches the ground, then hold the default pose for a moment
pose = robot.data.default_root_pose.torch.clone(); pose[:, 2] = 3.0; robot.write_root_pose_to_sim(pose)
q0 = robot.data.default_joint_pos.torch.clone(); robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
tgt = q0.clone(); jidx = torch.arange(N, device=q0.device) % J; tgt[torch.arange(N), jidx] += DELTA
for i in range(K):
    robot.set_joint_position_target(tgt); robot.write_data_to_sim(); NexusManager.step(); robot.update(env.physics_dt)
dq = robot.data.joint_pos.torch - q0                                  # (N, J)
own = dq[torch.arange(N), jidx]; others = dq.clone(); others[torch.arange(N), jidx] = 0.0
names = robot.joint_names; bad = []
for e in range(J):
    j = int(jidx[e]); o = float(own[e]); oth = float(others[e].abs().max()); k = int(others[e].abs().argmax())
    flag = "OK" if (o > 0.3 * DELTA and oth < 0.5 * abs(o)) else ("WRONG SIGN" if o < -0.05 else ("NO MOTION" if abs(o) < 0.05 else "CROSSTALK"))
    if flag != "OK": bad.append(names[j])
    print(f"  {names[j]:<28} moved {o:+.3f} rad toward +{DELTA} | largest other joint {names[k]} {oth:+.3f} | {flag}")
print(f"torque-direction test: {J - len(bad)}/{J} joints OK; bad: {bad}")
env.close(); app.close()
