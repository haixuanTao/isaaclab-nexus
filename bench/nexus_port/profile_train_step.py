"""Section timings inside a REAL rsl_rl training loop on the Nexus backend.

usage: profile_train_step.py [num_envs] [iterations]
Unlike profile_env_step.py this runs the actual PPO loop: real actions, the
DelayedDCMotor actuator, resets and every manager term.
"""
import os, sys, time
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym, torch
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
from agile.rl_env.rsl_rl import RslRlVecEnvWrapper, make_rsl_rl_runner
from isaaclab_nexus.envs import nexusify
from isaaclab_nexus.physics.nexus_manager import NexusManager

TASK = "HeightTracking-G1-v0"; G1 = os.environ.get("NEXUS_G1_MJCF", "/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 6

env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
env_cfg.scene.num_envs = N; env_cfg.seed = agent_cfg.seed
agent_cfg.max_iterations = ITERS; agent_cfg.run_name = "profile"
nexusify(env_cfg, G1)
env = gym.make(TASK, cfg=env_cfg)
base = env.unwrapped
robot = base.scene.articulations["robot"]

acc = {}
def timed(fn, key):
    acc.setdefault(key, 0.0)
    def wrap(*a, **kw):
        torch.cuda.synchronize(); t = time.perf_counter()
        out = fn(*a, **kw)
        torch.cuda.synchronize(); acc[key] += time.perf_counter() - t
        return out
    return wrap

NexusManager.step = staticmethod(timed(NexusManager.step, "physics"))
robot.write_data_to_sim = timed(robot.write_data_to_sim, "write+actuators")
robot.update = timed(robot.update, "articulation.update")
for s in base.scene.sensors.values():
    s.update = timed(s.update, "sensors")
base.action_manager.process_action = timed(base.action_manager.process_action, "action_mgr")
base.action_manager.apply_action = timed(base.action_manager.apply_action, "action_mgr")
base.observation_manager.compute = timed(base.observation_manager.compute, "obs_mgr")
base.reward_manager.compute = timed(base.reward_manager.compute, "reward_mgr")
base.termination_manager.compute = timed(base.termination_manager.compute, "termination_mgr")
base.event_manager.apply = timed(base.event_manager.apply, "event_mgr")
base._reset_idx = timed(base._reset_idx, "reset_idx TOTAL")
robot.write_joint_state_to_sim = timed(robot.write_joint_state_to_sim, "  reset: joint state write")
robot.write_root_pose_to_sim = timed(robot.write_root_pose_to_sim, "  reset: root pose write")
robot.write_root_velocity_to_sim = timed(robot.write_root_velocity_to_sim, "  reset: root vel write")
robot.reset = timed(robot.reset, "  reset: articulation.reset")
base.event_manager.reset = timed(base.event_manager.reset, "  reset: event_mgr.reset")
base.observation_manager.reset = timed(base.observation_manager.reset, "  reset: obs_mgr.reset")
base.curriculum_manager.compute = timed(base.curriculum_manager.compute, "  reset: curriculum")
base.step = timed(base.step, "TOTAL env.step")

env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
log_dir = os.path.abspath("/workspace/bench/nexus_port/logs/profile")
runner = make_rsl_rl_runner(env, agent_cfg, log_dir=log_dir, device=agent_cfg.device)
torch.cuda.synchronize(); t0 = time.perf_counter()
runner.learn(num_learning_iterations=ITERS, init_at_random_ep_len=True)
torch.cuda.synchronize(); wall = time.perf_counter() - t0

steps = ITERS * agent_cfg.num_steps_per_env
env_step = acc.pop("TOTAL env.step")
inner = sum(acc.values())
print(f"\n=== Nexus training-loop breakdown | {N} envs x {ITERS} iters x {agent_cfg.num_steps_per_env} steps ===")
print(f"{'section':<22} {'total_s':>9} {'ms/ctrl-step':>13} {'% of iter':>10}")
for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    print(f"{k:<22} {v:9.2f} {v/steps*1e3:13.2f} {v/wall*100:9.1f}%")
print(f"{'  (env.step, other)':<22} {env_step-inner:9.2f} {(env_step-inner)/steps*1e3:13.2f} {(env_step-inner)/wall*100:9.1f}%")
print(f"{'env.step TOTAL':<22} {env_step:9.2f} {env_step/steps*1e3:13.2f} {env_step/wall*100:9.1f}%")
print(f"{'PPO + policy + rest':<22} {wall-env_step:9.2f} {(wall-env_step)/steps*1e3:13.2f} {(wall-env_step)/wall*100:9.1f}%")
print(f"{'WALL':<22} {wall:9.2f} {wall/steps*1e3:13.2f} {100.0:9.1f}%  -> {N*steps/wall:.0f} env-steps/s")
env.close(); app.close()
