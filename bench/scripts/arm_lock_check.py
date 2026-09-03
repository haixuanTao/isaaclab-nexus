#!/usr/bin/env python3
"""Are the arms locked by the physics or held still by the policy?
Replays a policy harness-free and reports, per joint: range of motion, mean speed,
range of the policy's commanded target, and the in-solver torque cap / damping on
that DOF averaged over the run."""
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
parser.add_argument("--checkpoint", type=str, required=True); parser.add_argument("--envs", type=int, default=16); parser.add_argument("--steps", type=int, default=750)
AppLauncher.add_app_launcher_args(parser); args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, numpy as np, torch
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_newton.physics import NewtonManager
from agile.rl_env.rsl_rl.export_pruning import prepare_training_only_actions_for_evaluation
cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.envs); cfg.seed = 42; prepare_training_only_actions_for_evaluation(cfg)
env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped
from agile.rl_env.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
wenv = RslRlVecEnvWrapper(env); obs, _ = wenv.reset()
policy = torch.jit.load(args_cli.checkpoint, map_location=u.device).eval()
robot = u.scene["robot"]; names = list(robot.joint_names); J = len(names)
s = NewtonManager._solver; mjw = s.mjw_model; jm = s.mjc_jnt_to_newton_dof.numpy(); m = NewtonManager.get_model()
labels = [l.rsplit("/", 1)[-1] for l in m.joint_label]; qd_start = m.joint_qd_start.numpy(); dofadr = mjw.jnt_dofadr.numpy()
# joint name -> (mjc jnt, mjc dof) for world 0
j2mj = {}
for jn in names:
    nj = [k for k, l in enumerate(labels) if l == jn][0]; d = int(qd_start[nj]); mj = [k for k in range(jm.shape[1]) if int(jm[0, k]) == d][0]; j2mj[jn] = (mj, int(dofadr[mj]))
def tt(x): return x.torch if hasattr(x, "torch") else x
qmin = torch.full((J,), 1e9, device=u.device); qmax = torch.full((J,), -1e9, device=u.device); tmin = qmin.clone(); tmax = qmax.clone()
speed = torch.zeros(J, device=u.device); cap_sum = np.zeros(J); damp_sum = np.zeros(J); n = 0
limits = tt(robot.data.joint_pos_limits)[0]
with torch.inference_mode():
    for step in range(args_cli.steps):
        po = obs["policy"] if hasattr(obs, "keys") else obs
        obs, rew, dones, extras = wenv.step(policy(po))
        q = tt(robot.data.joint_pos); qd = tt(robot.data.joint_vel); tg = tt(robot.data.joint_pos_target)
        qmin = torch.minimum(qmin, q.min(0).values); qmax = torch.maximum(qmax, q.max(0).values)
        tmin = torch.minimum(tmin, tg.min(0).values); tmax = torch.maximum(tmax, tg.max(0).values)
        speed += qd.abs().mean(0); n += 1
        fr = mjw.jnt_actfrcrange.numpy()[0]; dd = mjw.dof_damping.numpy()[0]
        for j, jn in enumerate(names):
            mj, md = j2mj[jn]; cap_sum[j] += fr[mj][1]; damp_sum[j] += dd[md]
print(f"\n[arm] {'joint':26s} {'range(rad)':>10s} {'target rng':>10s} {'mean|qd|':>9s} {'mean cap':>9s} {'mean damp':>9s} {'limits':>16s}")
for j, jn in enumerate(names):
    flag = ""
    rng = float(qmax[j] - qmin[j]); trg = float(tmax[j] - tmin[j])
    if trg > 0.3 and rng < 0.05: flag = "  <-- target moves, joint does not (PHYSICS?)"
    elif trg < 0.1 and rng < 0.05: flag = "  <-- policy holds it still"
    print(f"[arm] {jn:26s} {rng:10.3f} {trg:10.3f} {float(speed[j]/n):9.2f} {cap_sum[j]/n:9.1f} {damp_sum[j]/n:9.3f} [{float(limits[j,0]):+.2f},{float(limits[j,1]):+.2f}]{flag}")
env.close(); simulation_app.close()
