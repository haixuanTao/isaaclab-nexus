#!/usr/bin/env python3
"""Does the RayCaster height sensor read the ground correctly on Newton rough terrain?
For each env: ray_hits_w.z (what the sensor thinks the ground is), the env-origin z and
the true terrain height under the robot, and base_height = body_z - ground. A broken
raycast shows as ray_hits_w.z = 0 / inf / far from the terrain surface."""
import argparse, sys, numpy as np
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(); p.add_argument("--task",default="HeightTracking-G1-v0"); p.add_argument("--envs",type=int,default=8); p.add_argument("--steps",type=int,default=30); p.add_argument("--level",type=int,default=6)
AppLauncher.add_app_launcher_args(p); a,hy=p.parse_known_args(); sys.argv=[sys.argv[0]]+hy
app=AppLauncher(a).app
import gymnasium as gym, torch
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
cfg=parse_env_cfg(a.task,num_envs=a.envs); env=gym.make(a.task,cfg=cfg); u=env.unwrapped
terr=u.scene.terrain
if getattr(terr,"terrain_levels",None) is not None: terr.terrain_levels[:]=min(a.level,int(terr.terrain_origins.shape[0])-1)
env.reset()
T=lambda x:x.torch if hasattr(x,"torch") else x
hs=u.scene.sensors["height_measurement_sensor"]
robot=u.scene["robot"]
act=torch.zeros(u.action_space.shape,device=u.device)
for step in range(a.steps): env.step(act)
rh=T(hs.data.ray_hits_w)                 # (E, R, 3)
gh=rh[...,2]                             # ground z per ray
origin_z=u.scene.env_origins[:,2]
body_z=T(robot.data.root_pos_w)[:,2]
finite=torch.isfinite(gh)
print(f"[hs] rays per env: {gh.shape[1]}  finite ray hits: {int(finite.sum())}/{gh.numel()} ({100*float(finite.float().mean()):.1f}%)")
print(f"[hs] ray_hits_w.z  : min={float(gh[finite].min()):+.3f} max={float(gh[finite].max()):+.3f} mean={float(gh[finite].mean()):+.3f}")
print(f"[hs] env_origin.z  : min={float(origin_z.min()):+.3f} max={float(origin_z.max()):+.3f} mean={float(origin_z.mean()):+.3f}")
print(f"[hs] robot root z  : min={float(body_z.min()):+.3f} max={float(body_z.max()):+.3f} mean={float(body_z.mean()):+.3f}")
gmean=torch.mean(torch.where(finite,gh,torch.zeros_like(gh)),dim=1)
base_h = body_z - gmean
print(f"[hs] per env: env_origin.z | mean ground(ray) | root z | base_height(root - ground):")
for e in range(min(a.envs,8)):
    print(f"[hs]   env {e}: origin={float(origin_z[e]):+.3f}  ground={float(gmean[e]):+.3f}  rootz={float(body_z[e]):+.3f}  base_h={float(base_h[e]):+.3f}")
# a healthy sensor: ground ~ env_origin.z +- 0.14 (terrain relief); base_h ~ 0.1-0.9 (a limp robot on the ground ~0.1-0.3)
off = float(torch.abs(gmean - origin_z).mean())
print(f"\n[hs] |mean ground - env_origin.z| avg = {off:.3f} m  -> {'SENSOR OK (tracks terrain)' if off < 0.3 else 'SENSOR WRONG (ground reading not on the terrain)'}", flush=True)
env.close(); app.close()
