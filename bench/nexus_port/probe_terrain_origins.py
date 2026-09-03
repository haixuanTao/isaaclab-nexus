"""Terrain origins (z) per (level, type) and each used tile's z range — do tiles of one level share an origin height?"""
import os, numpy as np
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym, torch
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_nexus.envs import nexusify
TASK = "HeightTracking-G1-v0"; env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); env_cfg.scene.num_envs = 256; env_cfg.seed = 7
nexusify(env_cfg, "/workspace/bench/nexus_port/g1_29dof_convex64.xml"); env = gym.make(TASK, cfg=env_cfg); base = env.unwrapped
T = base.scene.terrain; terr = T.terrain; o = np.asarray(terr.terrain_origins)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
print("terrain_origins z [level, type]:\n", o[..., 2]); print("terrain_origins xy of (0,0),(0,1):", o[0, 0, :2], o[0, 1, :2])
print("sub-terrain cfg:", [(k, type(v).__name__, getattr(v, 'proportion', None)) for k, v in env_cfg.scene.terrain.terrain_generator.sub_terrains.items()])
lv, ty = T.terrain_levels.cpu().numpy(), T.terrain_types.cpu().numpy(); print("levels in use:", np.unique(lv, return_counts=True), "| types:", np.unique(ty, return_counts=True))
for (r, c), V in sorted(terr.tile_vertices.items()): print(f"  tile L{r} T{c}: z range [{V[:,2].min():+.3f}, {V[:,2].max():+.3f}] origin z {o[r, c, 2]:+.3f} | height at centre {float(terr.height[terr._tile_index[(r,c)]][terr.height.shape[1]//2, terr.height.shape[2]//2]):+.3f}")
env.close(); app.close()
