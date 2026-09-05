"""Does the off-tile apron hold a robot? AGILE env on Nexus, lift off, zero actions: teleport robots to xy=(6,0)
(2 m outside the 8x8 tile, on the apron), (3.9,0) (tile edge) and (0,0) (tile centre) at z=1.3 and watch root z."""
import os, torch
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym
import agile.rl_env.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_nexus.envs import nexusify
TASK = "HeightTracking-G1-v0"; env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
env_cfg.scene.num_envs = 6; env_cfg.seed = 3; env_cfg.actions.lift.stiffness_forces = 0.0; env_cfg.actions.lift.damping_forces = 0.0; env_cfg.actions.lift.damping_torques = 0.0
for k in list(vars(env_cfg.events).keys()):
    if k.startswith("push") or k.startswith("apply_external") or "reset_from" in k or "fallen" in k: setattr(env_cfg.events, k, None)
nexusify(env_cfg, "/workspace/bench/nexus_port/g1_29dof_convex64.xml", agent_cfg=agent_cfg)
env = gym.make(TASK, cfg=env_cfg); base = env.unwrapped; robot = base.scene.articulations["robot"]; base.reset()
terr = base.scene.terrain.terrain; print("floor_half", getattr(terr, "_floor_half", "?"), "| apron bodies inserted:", "aprons" in dir(terr) or "n/a", "| tile_zmin:", {k: round(v, 3) for k, v in list(terr.tile_zmin.items())[:3]})
xy = torch.tensor([[6.0, 0.0], [-6.0, 0.0], [3.9, 0.0], [0.0, 3.9], [0.0, 0.0], [10.0, 0.0]], device=base.device)
pose = robot.data.default_root_state.torch[:, :7].clone(); pose[:, :2] = xy; pose[:, 2] = 1.3
robot.write_root_pose_to_sim(pose); robot.write_root_velocity_to_sim(torch.zeros(6, 6, device=base.device))
zero = torch.zeros(6, base.action_manager.total_action_dim, device=base.device); zs = []
with torch.no_grad():
    for i in range(150):
        base.step(zero); zs.append(robot.data.root_link_pos_w.torch[:, 2].clone())
zs = torch.stack(zs).cpu().numpy()
for e, tag in enumerate(("apron x=+6", "apron x=-6", "tile edge x=3.9", "tile edge y=3.9", "tile centre", "apron x=+10")):
    print(f"{tag:16s}: root z at 0.5/1/2/3 s = {zs[24, e]:+.2f} / {zs[49, e]:+.2f} / {zs[99, e]:+.2f} / {zs[149, e]:+.2f}")
env.close(); app.close()
