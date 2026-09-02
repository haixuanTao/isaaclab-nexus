#!/usr/bin/env python3
"""Per-link mass, COM and inertia tensor, by body name. Dumped to JSON for a
cross-engine diff. Wrong inertia = same torque, different acceleration."""
import argparse, sys, json
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, torch
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
def tt(x): return x.torch if hasattr(x, "torch") else x
cfg = parse_env_cfg(args_cli.task, num_envs=2); env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; robot = u.scene["robot"]
env.reset()
out = {"body_names": list(robot.body_names)}
for f in ("default_mass", "default_inertia", "body_com_pos_b", "default_joint_armature", "default_joint_friction_coeff",
          "joint_armature", "joint_friction_coeff"):
    v = tt(getattr(robot.data, f, None))
    out[f] = v[0].detach().cpu().tolist() if isinstance(v, torch.Tensor) and v.numel() else None
    print(f"[in] {f:28s} {'shape='+str(tuple(v.shape)) if isinstance(v, torch.Tensor) else 'n/a'}")
out["joint_names"] = list(robot.joint_names)
json.dump(out, open(args_cli.out, "w")); print(f"[in] wrote {args_cli.out}")
env.close(); simulation_app.close()
