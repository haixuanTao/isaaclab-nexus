#!/usr/bin/env python3
"""Dump the GPU MuJoCo geom contact parameters Newton built: friction, solref,
solimp, margin, gap, priority, condim -- for the feet and the terrain."""
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); parser.add_argument("--task", type=str, default="HeightTracking-G1-v0")
AppLauncher.add_app_launcher_args(parser); args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app
import gymnasium as gym, torch, numpy as np, warp as wp, gc, mujoco
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
cfg = parse_env_cfg(args_cli.task, num_envs=2); env = gym.make(args_cli.task, cfg=cfg); u = env.unwrapped; env.reset()
solver = next((o for o in gc.get_objects() if type(o).__name__ == "SolverMuJoCo"), None); mj = solver.mj_model; mw = solver.mjw_model
def g(name):
    a = getattr(mw, name, None)
    try: return wp.to_torch(a).float().cpu().numpy() if isinstance(a, wp.array) else np.asarray(a)
    except Exception: return None
names = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom{i}" for i in range(mj.ngeom)]
A = lambda n: np.asarray(getattr(mj, n))
fr = A("geom_friction"); sr = A("geom_solref"); si = A("geom_solimp"); mg = A("geom_margin"); gp = A("geom_gap"); cd = A("geom_condim"); pr = A("geom_priority")
def row(i, tag):
    e = 0
    def pick(a): 
        if a is None: return "n/a"
        return np.round(a[i], 4).tolist()
    print(f"[cp] {tag:8s} {names[i][-40:]:40s} friction={pick(fr)} solref={pick(sr)} solimp={pick(si)} margin={pick(mg)} gap={pick(gp)} condim={pick(cd)} priority={pick(pr)}")
feet = [i for i, n in enumerate(names) if "ankle_roll" in n or "foot" in n.lower()][:2]
terr = [i for i, n in enumerate(names) if "terrain" in n.lower() or "ground" in n.lower()][:1]
print(f"\n[cp] ngeom={mj.ngeom}  opt.o_margin/o_solref/o_solimp={getattr(mj.opt,'o_margin',None)} {np.asarray(getattr(mj.opt,'o_solref',[]))} {np.asarray(getattr(mj.opt,'o_solimp',[]))}  opt.impratio={mj.opt.impratio} opt.cone={mj.opt.cone}")
for i in feet: row(i, "FOOT")
for i in terr: row(i, "TERRAIN")
print(f"[cp] global: friction min/max over geoms = {fr.min():.3f}/{fr.max():.3f}; margin max={mg.max():.4f}; gap max={gp.max():.4f}; solref rows unique={len(np.unique(np.round(sr.reshape(-1, sr.shape[-1]),4),axis=0))}")
env.close(); simulation_app.close()
