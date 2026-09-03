"""Is the first observation after a reset consistent with the reset state, in the real AGILE env on Nexus?
Compares the obs Isaac computes at reset time with a recomputation after Articulation.update()
(which syncs and refreshes every derived buffer) with NO physics step in between."""
import os, torch
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym, importlib
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_nexus.envs import nexusify
TASK = "HeightTracking-G1-v0"; G1 = os.environ.get("NEXUS_G1_MJCF", "/workspace/bench/nexus_port/g1_29dof_convex64.xml"); N = 512
cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point"); cfg.scene.num_envs = N; cfg.seed = 42
nexusify(cfg, G1); env = gym.make(TASK, cfg=cfg).unwrapped
pre = gym.spec(TASK).kwargs.get("pre_learn_entry_point"); mod, fn = pre.split(":"); getattr(importlib.import_module(mod), fn)(env, TASK, agent_cfg)
robot = env.scene.articulations["robot"]; om = env.observation_manager
act = torch.zeros(N, env.action_manager.total_action_dim, device="cuda:0")
for _ in range(60): env.step(act)                                   # get into a mid-episode state
obs_at_reset, _ = env.reset()                                        # full reset -> obs computed by Isaac's path (after sim.forward())
robot.update(env.physics_dt)                                         # sync + refresh derived buffers, no physics step
for s in env.scene.sensors.values(): s.update(env.physics_dt)
obs_fresh = om.compute()
for grp in ("policy", "critic"):
    a = obs_at_reset[grp] if isinstance(obs_at_reset, dict) else obs_at_reset[grp]; b = obs_fresh[grp]
    a = a if torch.is_tensor(a) else torch.cat([t.flatten(1) for t in a.values()], 1); b = b if torch.is_tensor(b) else torch.cat([t.flatten(1) for t in b.values()], 1)
    d = (a - b).abs(); names = om.active_terms[grp]; dims = [int(torch.tensor(x).prod()) for x in om.group_obs_term_dim[grp]]; o = 0; bad = []
    for n, k in zip(names, dims):
        dd = d[:, o:o + k]; o += k
        if float(dd.max()) > 1e-3: bad.append(f"{n}: max diff {float(dd.max()):.3f} (frac envs >1e-3: {float((dd.max(1).values > 1e-3).float().mean()):.2f})")
    print(f"[{grp}] terms whose reset-time obs differ from the fresh recomputation: {bad if bad else 'NONE'}")
env.close(); app.close()
