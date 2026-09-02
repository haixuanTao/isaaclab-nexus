#!/usr/bin/env python3
"""Same root pose + same joint angles on each engine: do the links land in the
same places?  Dumps every body position relative to the root so the two runs can
be diffed. A joint whose axis/sign/offset differs between engines shows up as a
link in a different place.
"""
import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import agile.isaaclab_extras.monkey_patches  # noqa: F401,E402
import agile.rl_env.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def tt(x):
    return x.torch if hasattr(x, "torch") else x


cfg = parse_env_cfg(args_cli.task, num_envs=2)
cfg.seed = 42
env = gym.make(args_cli.task, cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
env.reset()

out = {"body_names": list(robot.body_names), "joint_names": list(robot.joint_names), "poses": {}}

def snapshot(tag):
    """Body positions relative to root, in the ROOT frame (so orientation cancels)."""
    u.scene.write_data_to_sim()
    u.sim.step()
    u.scene.update(u.physics_dt)
    rp = tt(robot.data.root_pos_w)[0]
    rq = tt(robot.data.root_quat_w)[0]          # (w,x,y,z)
    bp = tt(robot.data.body_pos_w)[0]           # (B,3)
    rel = bp - rp
    # rotate into root frame: q^-1 * v
    w, x, y, z = rq.tolist()
    import math
    def qinv_apply(v):
        vx, vy, vz = v
        # conjugate quaternion (w,-x,-y,-z) applied to v
        qx, qy, qz, qw = -x, -y, -z, w
        # t = 2 * cross(q.xyz, v)
        tx = 2*(qy*vz - qz*vy); ty = 2*(qz*vx - qx*vz); tz = 2*(qx*vy - qy*vx)
        return (vx + qw*tx + (qy*tz - qz*ty),
                vy + qw*ty + (qz*tx - qx*tz),
                vz + qw*tz + (qx*ty - qy*tx))
    rel_b = [qinv_apply(v.tolist()) for v in rel]
    jp = tt(robot.data.joint_pos)[0].tolist()
    out["poses"][tag] = {
        "root_pos_w": rp.tolist(), "root_quat_w": rq.tolist(),
        "joint_pos": jp, "body_rel_root_frame": rel_b,
    }
    print(f"[fk] {tag}: root_quat_w={[round(v,3) for v in rq.tolist()]}  "
          f"joint_pos[:6]={[round(v,3) for v in jp[:6]]}")

# Pose A: default joint positions, identity root, in the air (no contact).
st = tt(robot.data.root_state_w).clone()
st[:, 2] += 3.0; st[:, 3] = 1.0; st[:, 4:7] = 0.0; st[:, 7:13] = 0.0
robot.write_root_state_to_sim(st)
jp0 = tt(robot.data.default_joint_pos).clone()
robot.write_joint_state_to_sim(position=jp0, velocity=torch.zeros_like(jp0))
snapshot("A_default_pose")

# Pose B: bend every joint +0.5 rad from default -> sign convention shows up.
robot.write_root_state_to_sim(st)
robot.write_joint_state_to_sim(position=jp0 + 0.5, velocity=torch.zeros_like(jp0))
snapshot("B_default_plus_0p5")

json.dump(out, open(args_cli.out, "w"))
print(f"[fk] wrote {args_cli.out}")
env.close()
simulation_app.close()
