#!/usr/bin/env python3
"""Does write_joint_state_to_sim(env_ids=SUBSET) take effect on each engine?
Writes a distinctive pose to half the envs, reads back all envs by name."""
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--label", type=str, default="engine")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, torch
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
def tt(x): return x.torch if hasattr(x, "torch") else x
cfg = parse_env_cfg(args_cli.task, num_envs=8); cfg.seed = 42
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; robot = u.scene["robot"]
env.reset(); names = robot.joint_names; J = robot.num_joints; dt = u.physics_dt
sub = torch.tensor([1, 3, 5, 7], device=u.device)
print(f"\n[sw] ===== {args_cli.label} =====  writing to env_ids={sub.tolist()} only")
# baseline: everyone at default
jp0 = tt(robot.data.default_joint_pos).clone(); robot.write_joint_state_to_sim(position=jp0, velocity=torch.zeros_like(jp0)); u.scene.update(dt)
# distinctive pose for the subset: knee = 1.0, elbow = 1.5, waist_yaw = 0.7 (all within limits)
pose = jp0[:len(sub)].clone()
for n, v in (("left_knee_joint", 1.0), ("right_knee_joint", 1.0), ("left_elbow_joint", 1.5), ("waist_yaw_joint", 0.7)):
    pose[:, names.index(n)] = v
for mode in ("index", "mask"):
    robot.write_joint_state_to_sim(position=jp0, velocity=torch.zeros_like(jp0)); u.scene.update(dt)
    if mode == "index":
        robot.write_joint_state_to_sim_index(position=pose, velocity=torch.zeros_like(pose), env_ids=sub)
    else:
        import warp as wp
        m = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device); m[sub] = True
        full = jp0.clone(); full[sub] = pose
        robot.write_joint_state_to_sim_mask(position=full, velocity=torch.zeros_like(full), env_mask=wp.from_torch(m))
    u.scene.update(dt)
    q = tt(robot.data.joint_pos)
    k = names.index("left_knee_joint"); e = names.index("left_elbow_joint"); w = names.index("waist_yaw_joint")
    print(f"[sw] via {mode:5s}: knee per env = {[round(float(v),2) for v in q[:,k]]}")
    print(f"[sw] via {mode:5s}: elbow per env= {[round(float(v),2) for v in q[:,e]]}   waist_yaw = {[round(float(v),2) for v in q[:,w]]}")
    ok_sub = bool((q[sub, k] - 1.0).abs().max() < 1e-3 and (q[sub, e] - 1.5).abs().max() < 1e-3)
    others = torch.tensor([0, 2, 4, 6], device=u.device)
    ok_oth = bool((q[others, k] - jp0[0, k]).abs().max() < 1e-3)
    print(f"[sw] via {mode:5s}: subset written correctly={ok_sub}   others untouched={ok_oth}   "
          f"{'OK' if ok_sub and ok_oth else '<<< SUBSET WRITE BROKEN'}")
    # and does it survive one physics step (i.e. did the SOLVER state get it, not just the buffer)?
    u.sim.step(); u.scene.update(dt); q2 = tt(robot.data.joint_pos)
    print(f"[sw] via {mode:5s}: after 1 physics step, subset knee = {[round(float(v),2) for v in q2[sub,k]]}  "
          f"({'held in solver' if (q2[sub,k]-1.0).abs().max() < 0.05 else '<<< NOT IN SOLVER STATE'})")
env.close(); simulation_app.close()
