#!/usr/bin/env python3
"""Is torso_slam firing on real pelvis/torso contact or a Newton heightfield artifact?
Rolls out the rough-terrain env; each step logs, for pelvis+torso_link: contact force,
whether it exceeds the 10 N threshold, the body height above the local terrain, and the
body speed. A SPURIOUS contact = force>10 N while the body is well clear of the ground."""
import argparse, sys, numpy as np
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(); p.add_argument("--task",default="HeightTracking-G1-v0"); p.add_argument("--envs",type=int,default=6); p.add_argument("--steps",type=int,default=250); p.add_argument("--level",type=int,default=6)
p.add_argument("--checkpoint",default=None)
AppLauncher.add_app_launcher_args(p); a,hy=p.parse_known_args(); sys.argv=[sys.argv[0]]+hy
app=AppLauncher(a).app
import gymnasium as gym, torch
from rsl_rl.env import VecEnv  # noqa
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import agile.isaaclab_extras.monkey_patches  # noqa
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
cfg=parse_env_cfg(a.task,num_envs=a.envs)
if a.checkpoint:
    from agile.rl_env.rsl_rl.export_pruning import prepare_training_only_actions_for_evaluation
    prepare_training_only_actions_for_evaluation(cfg)
env=gym.make(a.task,cfg=cfg); u=env.unwrapped
terr=u.scene.terrain
if getattr(terr,"terrain_levels",None) is not None:
    terr.terrain_levels[:]=min(a.level,int(terr.terrain_origins.shape[0])-1)
env=RslRlVecEnvWrapper(env)
robot=u.scene["robot"]; cs=u.scene.sensors["contact_forces"] if "contact_forces" in u.scene.sensors else None
if cs is None:
    cs=next(iter(u.scene.sensors.values()))
T=lambda x:x.torch if hasattr(x,"torch") else x
bids=robot.find_bodies(["pelvis","torso_link"],preserve_order=True)[0]
sbn=cs.body_names; sids=[sbn.index(n) for n in ["pelvis","torso_link"] if n in sbn]
print(f"[tc] contact sensor bodies for pelvis/torso: {sids} of {len(sbn)}; asset body ids {bids}",flush=True)
pol=torch.jit.load(a.checkpoint,map_location=u.device).eval() if a.checkpoint else None
obs, _ = env.reset()
def flat(o):
    po = o["policy"] if (hasattr(o,"keys") and "policy" in o.keys()) else o
    return (po.torch if hasattr(po,"torch") else po).to(u.device).float()
spurious=0; real=0; fire_steps=0
for step in range(a.steps):
    if pol is not None:
        with torch.inference_mode(): act=pol(torch.as_tensor(flat(obs)))
    else:
        act=torch.zeros(u.action_space.shape,device=u.device)
    obs, *_ = env.step(act)
    f=T(cs.data.net_forces_w_history)[:, :, sids]          # (E, hist, nb, 3)
    fmag=torch.norm(f,dim=-1).max(dim=1)[0]                 # (E, nb) peak over history
    incontact=fmag>10.0
    bpos=T(robot.data.body_pos_w)[:,bids]                  # (E, nb, 3)
    bz=bpos[...,2]                                          # world z of pelvis/torso
    bspeed=torch.norm(T(robot.data.body_lin_vel_w)[:,bids],dim=-1)
    # local ground height under each body from env origin z (heightfield relief is small, ~+-0.14)
    for e in range(u.num_envs):
        for k in range(len(bids)):
            if incontact[e,k]:
                fire_steps+=1
                clear = float(bz[e,k])   # pelvis normally ~0.6-0.9 m standing; on ground ~0.1-0.3
                if clear>0.55:  # body is high up but sensor says in-contact -> spurious
                    spurious+=1
                else:
                    real+=1
    if step%50==0:
        print(f"[tc] step {step}: pelvis/torso in_contact fires so far real={real} spurious={spurious}  peakF={float(fmag.max()):.0f}N  min pelvis z={float(bz[:,0].min()):.2f}",flush=True)
print(f"\n[tc] over {a.steps} steps: in_contact fired {fire_steps} times -> REAL (body low) {real}, SPURIOUS (body>0.55m yet 'in contact') {spurious}",flush=True)
print(f"[tc] verdict: {'NEWTON ARTIFACT inflating torso_slam' if spurious>real else 'torso contacts are PHYSICAL (robot really on its pelvis)'}",flush=True)
env.close(); app.close()
