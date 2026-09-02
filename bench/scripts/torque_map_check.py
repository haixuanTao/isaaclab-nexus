#!/usr/bin/env python3
"""Apply torque to ONE named joint; report which joint actually accelerates.

Positions are proven to map by name (FK identical). Efforts use a separate
binding. If torquing 'left_knee' moves some other joint on Newton, the effort
index map is wrong -- and every policy action lands on the wrong limb.
"""
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--label", type=str, default="engine")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, torch
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
def tt(x): return x.torch if hasattr(x, "torch") else x

cfg = parse_env_cfg(args_cli.task, num_envs=4); cfg.seed = 42
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; robot = u.scene["robot"]
env.reset()
names = robot.joint_names
# PD off so the only torque is the one we inject; keep effort limits so it passes through clipped.
for act in robot.actuators.values():
    for a in ("stiffness", "damping"):
        v = getattr(act, a, None)
        if isinstance(v, torch.Tensor): v.zero_()

def fresh():
    st = tt(robot.data.root_state_w).clone()
    st[:, 2] += 5.0; st[:, 3] = 1.0; st[:, 4:7] = 0.0; st[:, 7:13] = 0.0
    robot.write_root_state_to_sim(st)
    jp = tt(robot.data.default_joint_pos).clone()
    robot.write_joint_state_to_sim(position=jp, velocity=torch.zeros_like(jp))
    u.scene.write_data_to_sim(); u.sim.step(); u.scene.update(u.physics_dt)

print(f"\n[tq] ===== {args_cli.label} =====")
for target in ["left_knee_joint", "right_elbow_joint", "waist_yaw_joint", "left_ankle_roll_joint"]:
    fresh()
    idx = names.index(target)
    tau = torch.zeros(u.num_envs, robot.num_joints, device=u.device)
    tau[:, idx] = 20.0
    for _ in range(4):                       # a few physics steps of pure torque
        robot.set_joint_effort_target(tau)
        u.scene.write_data_to_sim(); u.sim.step(); u.scene.update(u.physics_dt)
    jv = tt(robot.data.joint_vel)[0].abs()
    order = torch.argsort(jv, descending=True)[:3].tolist()
    moved = names[order[0]]
    print(f"[tq] torqued {target:24s} -> moved most: {moved:24s} "
          f"|vel|={float(jv[order[0]]):6.2f}   next: {names[order[1]]}={float(jv[order[1]]):.2f}, "
          f"{names[order[2]]}={float(jv[order[2]]):.2f}   {'OK' if moved == target else '<<< WRONG JOINT'}")
env.close(); simulation_app.close()
