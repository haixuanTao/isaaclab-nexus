"""Print the G1's per-joint velocity/effort limits as PhysX (Isaac Lab USD) reports them."""
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym, torch, json
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
cfg = load_cfg_from_registry("HeightTracking-G1-v0", "env_cfg_entry_point"); cfg.scene.num_envs = 2
env = gym.make("HeightTracking-G1-v0", cfg=cfg); r = env.unwrapped.scene.articulations["robot"]
vl = r.data.joint_velocity_limits; vl = vl.torch if hasattr(vl, "torch") else vl; el = r.data.joint_effort_limits; el = el.torch if hasattr(el, "torch") else el
print("PHYSX_JOINT_LIMITS " + json.dumps({n: [round(float(v), 2), round(float(e), 2)] for n, v, e in zip(r.joint_names, vl[0], el[0])}))
env.close(); app.close()
